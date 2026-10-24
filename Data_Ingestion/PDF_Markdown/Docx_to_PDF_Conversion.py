import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union


def _find_libreoffice_binary() -> str:
    """Locate the LibreOffice executable across Linux, macOS, and Windows."""
    for cmd in ("libreoffice", "soffice"):
        found = shutil.which(cmd)
        if found:
            return found

    system = platform.system()
    fallback_paths = []
    if system == "Darwin":
        fallback_paths = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]
    elif system == "Windows":
        fallback_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
    else:
        fallback_paths = [
            "/usr/bin/libreoffice",
            "/usr/bin/soffice",
            "/snap/bin/libreoffice",
        ]

    for path in fallback_paths:
        if Path(path).is_file():
            return path

    raise FileNotFoundError(
        "LibreOffice executable ('libreoffice' or 'soffice') was not found. "
        "Ensure LibreOffice is installed and available on your system PATH."
    )


def convert_docx_to_pdf(
    docx_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    timeout: int = 60,
) -> str:
    """Convert a .docx file to a .pdf file using LibreOffice in headless mode.

    Args:
        docx_path: Path to the source .docx file.
        output_dir: Directory where the converted PDF will be saved.
            Defaults to the same directory as `docx_path`.
        timeout: Maximum time in seconds to wait for conversion.

    Returns:
        The absolute file path to the generated .pdf file as a string.

    Raises:
        FileNotFoundError: If `docx_path` or LibreOffice binary does not exist,
            or if the expected PDF file was not created.
        ValueError: If the input file is not a .docx file.
        RuntimeError: If the LibreOffice conversion process fails or times out.
    """
    input_file = Path(docx_path).expanduser().resolve()

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if input_file.suffix.lower() != ".docx":
        raise ValueError(f"Expected a .docx file, got: {input_file.name}")

    target_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else input_file.parent
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    soffice_bin = _find_libreoffice_binary()

    # Use a temporary user profile so headless runs do not conflict with open GUI instances
    with tempfile.TemporaryDirectory(prefix="lo_profile_") as profile_dir:
        profile_uri = Path(profile_dir).as_uri()
        command = [
            soffice_bin,
            f"-env:UserInstallation={profile_uri}",
            "--headless",
            "--nologo",
            "--nodefault",
            "--nofirststartwizard",
            "--convert-to",
            "pdf:writer_pdf_Export",
            "--outdir",
            str(target_dir),
            str(input_file),
        ]

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"LibreOffice conversion timed out after {timeout} seconds."
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"LibreOffice failed with exit code {result.returncode}.\n"
                f"STDOUT: {result.stdout.strip()}\n"
                f"STDERR: {result.stderr.strip()}"
            )

    pdf_path = target_dir / f"{input_file.stem}.pdf"
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Conversion completed without errors, but output PDF was not found "
            f"or is empty at: {pdf_path}"
        )

    return str(pdf_path)


if __name__ == "__main__":
    

    src_docx = "/home/pankaj/projects/Agentic_Rag/Pankaj_Sharma_Resume.docx"
    out_folder = "/home/pankaj/projects/Agentic_Rag/Data_Ingestion/Docling_PDF_Parser/Output"

    try:
        generated_pdf = convert_docx_to_pdf(src_docx, out_folder)
        print(f"PDF generated at: {generated_pdf}")
    except Exception as err:
        print(f"Error: {err}")
        