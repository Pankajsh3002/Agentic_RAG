from Model_Calling.Model_Service import get_llm, build_image_message

llm = get_llm("default")

msg = build_image_message(
    prompt="Extract the key points from these resume.",
    file_path="Pankaj_Sharma_AI_Engineer.pdf",
    pages="1,2,3",
)
print(llm.invoke([msg]).content)