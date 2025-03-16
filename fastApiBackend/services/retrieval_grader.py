"""
This module implements a retrieval grading pipeline that evaluates the relevance of retrieved documents to a user's query. The grader ensures that only relevant documents are processed further, filtering out erroneous retrievals based on semantic meaning and keyword matching.
"""
from services.llm_instance import json_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide output as a JSON with two keys: 'score' for the binary score yes or no, and a key 'confidence' that indicate the score of yes between 0 and 1.
    """
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | json_llm | JsonOutputParser()
#
# question = "LLM agent memory"
# docs = retriever.invoke(question)
# doc_txt = docs[0].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
#
