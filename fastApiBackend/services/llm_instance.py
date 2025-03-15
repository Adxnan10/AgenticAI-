from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "llama3.2",
    temperature = 0.8,
    num_predict = 256, # Max number of tokens to generate. I set it to 256 but that is not the limit
    format="json"
)

