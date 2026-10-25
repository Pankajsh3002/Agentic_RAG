from Model_Calling.embedder import embed_texts
from Pipeline.chunker import chunk_markdown
from db.database import save_chunks


def run(doc_id: str, markdown: str) -> None:
    chunks = chunk_markdown(markdown)
    if not chunks:
        raise ValueError("No chunks produced from markdown")
    vectors = embed_texts([c["embed_text"] for c in chunks])
    save_chunks(doc_id, chunks, vectors)