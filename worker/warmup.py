import sys

from Data_Ingestion.PDF_Markdown.Parse_PDF import parse_pdf_with_docling

markdown = parse_pdf_with_docling("/opt/warmup/warmup.pdf", max_workers=1)
if not markdown or not markdown.strip():
    sys.exit("Warm-up failed: the sample PDF produced no text")
print("Warm-up OK:", len(markdown), "characters")