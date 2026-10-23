import fitz

from Model_Calling.Model_Service import get_llm, invoke_llm



def parse_pdf_with_llm(pdf_path: str) -> str:
    """
    Parses a PDF using LLM and extracts OCR text from images.

    Args:
        pdf_path (str): Path to the PDF file.
        local_models_path (str): Path to the local LLM models.

    Returns:
        str: The parsed content as a markdown string.
    """
    with fitz.open(pdf_path) as fitz_doc:
        total_pages = len(fitz_doc)

    complete_text = ""

    for page_number in range(1, total_pages + 1):
    
        vlm_response = invoke_llm( model_name="openai/gpt-6-luna",
                                            prompt="Extract the text exactly as it appears in the image and format as Markdown.",
                                            is_image="y",                    # case-insensitive
                                            image_path=pdf_path,
                                            pages=page_number,
                                            )

        # print(vlm_response)
        
        extracted_text = vlm_response.get("response", "").strip()

        complete_text += f"{extracted_text}\n\nPage {page_number} of {total_pages}\n\n"

    return complete_text


if __name__ == "__main__":
    pdf_path = "/home/pankaj/projects/Agentic_Rag/test.pdf"
    parsed_content = parse_pdf_with_llm(pdf_path)
    print(parsed_content)