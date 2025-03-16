"""
This module implements a retrieval grading pipeline that evaluates the relevance of retrieved documents to a user's query. The grader ensures that only relevant documents are processed further, filtering out erroneous retrievals based on semantic meaning and keyword matching.
"""
from services.llm_instance import json_llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal


class GradeRelevance(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        ..., description="Binary score: 'yes' means the document is relevant, 'no' means it is not."
    )
    confidence: float = Field(
        ..., description="Confidence score for the 'yes' decision, as a number between 0 and 1."
    )


# Wrap the LLM with structured output using the Pydantic model.
structured_llm_retrieval_grader = json_llm.with_structured_output(GradeRelevance)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Be  lenient in your scoring—lean towards 'yes' when there is medium indication of relevance.\n 
    Provide output as a JSON with two keys: 'binary_score' for the binary score yes or no, and a key 'confidence' that indicate the score of yes between 0.1 and 1. \n
    Return your answer strictly as a JSON object with two keys: 'binary_score' and 'confidence'.
    """
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_retrieval_grader


