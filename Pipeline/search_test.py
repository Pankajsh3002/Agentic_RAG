import sys

from Model_Calling.embedder import embed_query
from db.database import search_chunks

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python3 -m Pipeline.search_test <workspace_id> "your question"')

    workspace_id = sys.argv[1]
    question = " ".join(sys.argv[2:]) or "What is this document about?"
    results = search_chunks(embed_query(question), workspace_id, limit=5)

    if not results:
        print("No chunks found in this workspace.")

    for r in results:
        print(f"{r['similarity']:.3f} | {r['filename']} | {r['heading_path']}")
        print(r["content"][:300].replace("\n", " "), "\n")