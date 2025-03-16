from pydantic import BaseModel, Field
from services.llm_instance import json_llm
from langchain_core.prompts import ChatPromptTemplate


class AnswerGrader(BaseModel):
    binary_score: str = Field(description="Answer addresses the question. 'yes' or 'no'")


structured_llm_grader = json_llm.with_structured_output(AnswerGrader)
# prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
# TODO: Do the same for other json outputs, and move all pydantic data models to another dir (pydantic_models)

# run
# question = "What is Agent Memory? "
# docs = retriever.invoke(question)
# context = format_docs(docs)
# generated_answer = rag_chain.invoke({"context": context, "question": question})
# print(generated_answer)
# print(answer_grader.invoke(input={"question": question, "generation": generated_answer}))
