"""
This module implements generation pipeline based on context (relevant docs)
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from services.llm_instance import llm

system = """You are a retrieval-augmented generator. \n
    You are provided with a context that may contain extraneous or unrelated information. \n
    Your task is to answer the following question directly, using only the parts of the context that are relevant. \n
    Do not repeat or comment on the context if it is not necessary for the answer. \n
    If you don't know the answer, say I don't know.
    """
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question} \n\n Now The context for this which is not the input is: \n\n {context} "
                  "\n\n"),
    ]
)
# Post-processing: Convert docs to a string
def format_docs(docs):
    return "\n\n".join(doc.page_content.replace("input", "").replace("Input", "") for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
# question = "Agent Memory"
# docs = retriever.invoke(question)
# # print(docs)
# context = format_docs(docs)  # Convert docs to a string context
# # print("Context:\n{}".format(context))
# print(rag_chain.invoke({"context": context, "question": question}))
