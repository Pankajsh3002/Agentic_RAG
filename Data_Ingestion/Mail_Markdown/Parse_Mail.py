import base64
import email
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from email import policy
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


def _build_converter() -> DocumentConverter:
    """Builds and returns a Docling DocumentConverter instance."""
    return DocumentConverter()


def _sanitize_filename(filename: str, fallback_index: int) -> str:
    """Ensures an attachment filename is safe to write to disk."""
    if not filename or not filename.strip():
        return f"attachment_{fallback_index}"
    clean_name = os.path.basename(filename.strip())
    for ch in ('/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0'):
        clean_name = clean_name.replace(ch, "_")
    return clean_name or f"attachment_{fallback_index}"


def _unique_output_path(output_dir: str, filename: str) -> str:
    """Returns a non-colliding file path inside output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    candidate = os.path.join(output_dir, filename)
    if not os.path.exists(candidate):
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = os.path.join(output_dir, f"{stem}_{counter}{suffix}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _convert_html_with_docling(converter: DocumentConverter, html_content: str) -> str:
    """
    Converts an HTML string to Markdown using Docling's DocumentConverter.
    Supports both newer Docling versions (convert_string) and older versions (temp .html file).
    """
    if not html_content or not html_content.strip():
        return ""

    if hasattr(converter, "convert_string"):
        try:
            result = converter.convert_string(html_content, format=InputFormat.HTML)
            return result.document.export_to_markdown().strip()
        except Exception as e:
            logger.debug("convert_string fallback to temp HTML file: %s", e)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", encoding="utf-8", delete=False
    ) as tmp_html:
        tmp_html_path = tmp_html.name
        tmp_html.write(html_content)

    try:
        result = converter.convert(tmp_html_path)
        return result.document.export_to_markdown().strip()
    finally:
        if os.path.exists(tmp_html_path):
            os.remove(tmp_html_path)


def _format_email_markdown(
    headers: Dict[str, str],
    body_markdown: str,
    saved_attachments: List[str],
) -> str:
    """
    Formats email metadata headers, parsed Markdown body, and extracted attachment paths
    into a clean, unified Markdown document.
    """
    lines: List[str] = []

    subject = headers.get("Subject") or "(No Subject)"
    lines.append(f"# {subject}")
    lines.append("")

    header_order = ["From", "To", "Cc", "Bcc", "Date", "Subject"]
    for key in header_order:
        val = headers.get(key)
        if val and str(val).strip():
            lines.append(f"- **{key}**: {str(val).strip()}")

    lines.append("")
    lines.append("## Body")
    lines.append("")
    if body_markdown and body_markdown.strip():
        lines.append(body_markdown.strip())
    else:
        lines.append("*(Empty email body)*")

    if saved_attachments:
        lines.append("")
        lines.append("## Attachments")
        lines.append("")
        for att_path in saved_attachments:
            att_name = os.path.basename(att_path)
            lines.append(f"- **{att_name}**: `{att_path}`")

    return "\n".join(lines).strip()


def _parse_eml(
    email_path: str,
    attachment_output_path: str,
    converter: DocumentConverter,
) -> str:
    """
    Parses an .eml (RFC 822 / MIME) email file:
    1. Extracts metadata headers and saves all attachments to attachment_output_path.
    2. Converts the email body (HTML or plain text) to Markdown using Docling.
    """
    with open(email_path, "rb") as f:
        msg: EmailMessage = email.message_from_binary_file(f, policy=policy.default)

    headers = {
        "From": str(msg.get("From", "")),
        "To": str(msg.get("To", "")),
        "Cc": str(msg.get("Cc", "")),
        "Bcc": str(msg.get("Bcc", "")),
        "Date": str(msg.get("Date", "")),
        "Subject": str(msg.get("Subject", "")),
    }

    saved_attachments: List[str] = []
    html_parts: List[str] = []
    text_parts: List[str] = []
    att_counter = 1

    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue

            content_disposition = part.get_content_disposition()
            content_type = part.get_content_type()
            filename = part.get_filename()

            if content_disposition == "attachment" or filename:
                payload = part.get_payload(decode=True)
                if payload is not None:
                    safe_name = _sanitize_filename(filename or "", att_counter)
                    att_counter += 1
                    dest_path = _unique_output_path(attachment_output_path, safe_name)
                    with open(dest_path, "wb") as out_f:
                        out_f.write(payload)
                    saved_attachments.append(dest_path)
                continue

            if content_type == "text/html":
                try:
                    html_parts.append(part.get_content())
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_parts.append(payload.decode("utf-8", errors="replace"))
            elif content_type == "text/plain":
                try:
                    text_parts.append(part.get_content())
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload:
                        text_parts.append(payload.decode("utf-8", errors="replace"))
    else:
        content_type = msg.get_content_type()
        try:
            content = msg.get_content()
        except Exception:
            payload = msg.get_payload(decode=True)
            content = payload.decode("utf-8", errors="replace") if payload else ""

        if content_type == "text/html":
            html_parts.append(content)
        else:
            text_parts.append(content)

    body_markdown = ""
    if html_parts:
        combined_html = "\n".join(html_parts)
        body_markdown = _convert_html_with_docling(converter, combined_html)
    elif hasattr(InputFormat, "EMAIL"):
        try:
            result = converter.convert(email_path)
            body_markdown = result.document.export_to_markdown().strip()
        except Exception as e:
            logger.debug("Native Docling EMAIL conversion fallback to plain text: %s", e)

    if not body_markdown and text_parts:
        body_markdown = "\n\n".join(t.strip() for t in text_parts if t and t.strip())

    return _format_email_markdown(headers, body_markdown, saved_attachments)


def _parse_msg(
    email_path: str,
    attachment_output_path: str,
    converter: DocumentConverter,
) -> str:
    """
    Parses a Microsoft Outlook .msg file:
    1. Extracts headers and saves attachments to attachment_output_path using extract_msg.
    2. Converts the HTML body (or plain text body) to Markdown using Docling.
    """
    try:
        import extract_msg
    except ImportError as exc:
        raise ImportError(
            "The 'extract-msg' package is required to parse Outlook .msg files. "
            "Install it via: pip install extract-msg"
        ) from exc

    msg = extract_msg.openMsg(email_path)
    try:
        headers = {
            "From": str(getattr(msg, "sender", "") or ""),
            "To": str(getattr(msg, "to", "") or ""),
            "Cc": str(getattr(msg, "cc", "") or ""),
            "Bcc": str(getattr(msg, "bcc", "") or ""),
            "Date": str(getattr(msg, "date", "") or ""),
            "Subject": str(getattr(msg, "subject", "") or ""),
        }

        saved_attachments: List[str] = []
        for idx, attachment in enumerate(getattr(msg, "attachments", []), start=1):
            raw_name = (
                getattr(attachment, "longFilename", None)
                or getattr(attachment, "shortFilename", None)
                or getattr(attachment, "name", None)
                or f"attachment_{idx}"
            )
            safe_name = _sanitize_filename(str(raw_name), idx)
            dest_path = _unique_output_path(attachment_output_path, safe_name)

            data = getattr(attachment, "data", None)
            if isinstance(data, (bytes, bytearray)):
                with open(dest_path, "wb") as out_f:
                    out_f.write(data)
                saved_attachments.append(dest_path)
            elif hasattr(attachment, "save"):
                os.makedirs(attachment_output_path, exist_ok=True)
                attachment.save(
                    customPath=attachment_output_path,
                    customFilename=os.path.basename(dest_path),
                )
                saved_attachments.append(dest_path)

        html_body: Optional[str] = None
        raw_html = getattr(msg, "htmlBody", None)
        if raw_html:
            if isinstance(raw_html, (bytes, bytearray)):
                html_body = raw_html.decode("utf-8", errors="replace")
            else:
                html_body = str(raw_html)

        body_markdown = ""
        if html_body and html_body.strip():
            body_markdown = _convert_html_with_docling(converter, html_body)

        if not body_markdown:
            plain_body = getattr(msg, "body", "") or ""
            if isinstance(plain_body, (bytes, bytearray)):
                plain_body = plain_body.decode("utf-8", errors="replace")
            body_markdown = str(plain_body).strip()

        return _format_email_markdown(headers, body_markdown, saved_attachments)
    finally:
        if hasattr(msg, "close"):
            msg.close()


def _local_tag_name(tag: str) -> str:
    """Strips XML namespace from a tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1].lower()
    return tag.lower()


def _parse_xml_email(
    email_path: str,
    attachment_output_path: str,
    converter: DocumentConverter,
) -> str:
    """
    Parses an XML-formatted email file:
    1. Extracts standard email fields (From, To, Cc, Bcc, Date, Subject, Body/HTMLBody)
       and any base64-encoded <Attachment> nodes to attachment_output_path.
    2. Converts HTML/XML body content into Markdown via Docling.
    """
    tree = ET.parse(email_path)
    root = tree.getroot()

    headers: Dict[str, str] = {
        "From": "",
        "To": "",
        "Cc": "",
        "Bcc": "",
        "Date": "",
        "Subject": "",
    }
    html_parts: List[str] = []
    text_parts: List[str] = []
    saved_attachments: List[str] = []
    att_counter = 1

    tag_map = {
        "from": "From",
        "sender": "From",
        "to": "To",
        "recipient": "To",
        "recipients": "To",
        "cc": "Cc",
        "bcc": "Bcc",
        "date": "Date",
        "sent": "Date",
        "sentdate": "Date",
        "subject": "Subject",
        "title": "Subject",
    }

    for elem in root.iter():
        tname = _local_tag_name(elem.tag)
        text_val = (elem.text or "").strip()

        if tname in tag_map and text_val and not headers[tag_map[tname]]:
            headers[tag_map[tname]] = text_val
        elif tname in ("htmlbody", "bodyhtml", "html"):
            if text_val:
                html_parts.append(text_val)
        elif tname in ("body", "textbody", "plaintext", "message", "content"):
            if text_val:
                if "<html" in text_val.lower() or "<table" in text_val.lower() or "<p>" in text_val.lower():
                    html_parts.append(text_val)
                else:
                    text_parts.append(text_val)
        elif tname == "attachment":
            att_name = elem.attrib.get("filename") or elem.attrib.get("name") or ""
            att_data_b64 = elem.attrib.get("content") or elem.attrib.get("data") or ""

            for child in elem:
                cname = _local_tag_name(child.tag)
                ctext = (child.text or "").strip()
                if cname in ("filename", "name") and ctext:
                    att_name = ctext
                elif cname in ("content", "data", "base64", "bytes") and ctext:
                    att_data_b64 = ctext

            if not att_data_b64 and text_val and len(list(elem)) == 0:
                att_data_b64 = text_val

            if att_data_b64:
                safe_name = _sanitize_filename(att_name, att_counter)
                att_counter += 1
                dest_path = _unique_output_path(attachment_output_path, safe_name)
                try:
                    raw_bytes = base64.b64decode(att_data_b64)
                except Exception:
                    raw_bytes = att_data_b64.encode("utf-8")
                with open(dest_path, "wb") as out_f:
                    out_f.write(raw_bytes)
                saved_attachments.append(dest_path)

    body_markdown = ""
    if html_parts:
        body_markdown = _convert_html_with_docling(converter, "\n".join(html_parts))
    elif text_parts:
        body_markdown = "\n\n".join(text_parts)
    else:
        try:
            result = converter.convert(email_path)
            body_markdown = result.document.export_to_markdown().strip()
        except Exception:
            xml_str = ET.tostring(root, encoding="unicode", method="xml")
            body_markdown = _convert_html_with_docling(converter, f"<pre>{xml_str}</pre>")

    return _format_email_markdown(headers, body_markdown, saved_attachments)


def parse_email_with_docling(email_path: str, attachment_output_path: str) -> str:
    """
    Parses an email file (.eml, .msg, or .xml) using Docling, saves any attachments
    to attachment_output_path, and returns the email content as Markdown.

    Args:
        email_path (str): Path to the input email file (.eml, .msg, or .xml).
        attachment_output_path (str): Directory path where extracted attachments will be saved.

    Returns:
        str: Formatted Markdown containing email metadata, Docling-converted body, and saved attachments.
    """
    if not os.path.isfile(email_path):
        raise FileNotFoundError(f"Email file not found: {email_path}")

    os.makedirs(attachment_output_path, exist_ok=True)
    ext = Path(email_path).suffix.lower()
    converter = _build_converter()

    if ext == ".eml":
        return _parse_eml(email_path, attachment_output_path, converter)
    elif ext == ".msg":
        return _parse_msg(email_path, attachment_output_path, converter)
    elif ext == ".xml":
        return _parse_xml_email(email_path, attachment_output_path, converter)
    else:
        raise ValueError(
            f"Unsupported email file extension '{ext}'. Expected '.eml', '.msg', or '.xml'."
        )


if __name__ == "__main__":
    sample_mail = "sample_email.eml"
    attachments_dir = "./extracted_attachments"
    if os.path.exists(sample_mail):
        markdown_content = parse_email_with_docling(sample_mail, attachments_dir)
        print("Final Email Markdown Preview")
        print(markdown_content)