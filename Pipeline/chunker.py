import re

MAX_CHARS = 1500
HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split by markdown headings. Returns [(heading_path, body_text), ...]."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []     # (heading level, title)
    buf: list[str] = []
    in_fence = False

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))
        buf.clear()

    for line in markdown.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence        # '#' inside code blocks is not a heading
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last resort for one huge line: cut at a space near the limit."""
    pieces = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def _pack(text: str, max_chars: int) -> list[str]:
    """Pack paragraphs (then lines, then words) into pieces of at most max_chars."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    current = ""

    def add(unit: str, sep: str):
        nonlocal current
        if current and len(current) + len(sep) + len(unit) > max_chars:
            pieces.append(current)
            current = ""
        current = f"{current}{sep}{unit}" if current else unit

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            add(para, "\n\n")
        else:
            for line in para.splitlines():
                line = line.strip()
                if not line:
                    continue
                if len(line) <= max_chars:
                    add(line, "\n")
                else:
                    for part in _hard_split(line, max_chars):
                        add(part, "\n")
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(markdown: str, max_chars: int = MAX_CHARS) -> list[dict]:
    chunks: list[dict] = []
    for heading_path, body in _split_sections(markdown):
        for piece in _pack(body, max_chars):
            piece = piece.strip()
            if not piece:
                continue
            heading = heading_path[:300]
            chunks.append({
                "chunk_index": len(chunks),
                "heading_path": heading_path,
                "content": piece,
                "embed_text": f"{heading}\n\n{piece}" if heading else piece,
            })
    return chunks


if __name__ == "__main__":
    # Inspect chunking for a stored document: python3 -m Pipeline.chunker <doc_id>
    import sys
    from db.database import get_conn

    row = get_conn().execute(
        "SELECT markdown FROM documents WHERE doc_id = %s", (sys.argv[1],)
    ).fetchone()
    chunks = chunk_markdown(row["markdown"])
    sizes = [len(c["content"]) for c in chunks]
    print(f"{len(chunks)} chunks | min {min(sizes)} | avg {sum(sizes)//len(sizes)} | max {max(sizes)}")
    for c in chunks[:3]:
        print("---", c["heading_path"])
        print(c["content"][:300])