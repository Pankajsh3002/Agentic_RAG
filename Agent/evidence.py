class Evidence:
    """Everything the tools returned while answering ONE question, kept with the
    file each piece came from. The verifier checks the answer against this."""

    def __init__(self):
        self.items: list[tuple[str, str]] = []

    def add(self, filename: str, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.items.append((filename or "", text))

    def all_text(self) -> str:
        return "\n".join(text for _, text in self.items)

    def text_for(self, filename: str) -> str:
        name = (filename or "").lower()
        return "\n".join(text for f, text in self.items if f.lower() == name)

    def render(self, max_chars: int) -> str:
        """Evidence grouped for a model to read, cut at max_chars."""
        blocks, size = [], 0
        for filename, text in self.items:
            block = f"### {filename or 'unknown'}\n{text}"
            if size + len(block) > max_chars:
                blocks.append("(more evidence was cut for length)")
                break
            blocks.append(block)
            size += len(block) + 2
        return "\n\n".join(blocks)