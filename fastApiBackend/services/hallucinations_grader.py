from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from services.llm_instance import json_llm
from pydantic import BaseModel, Field
from typing import Literal
from services.generate_from_docs import format_docs, rag_chain
from services.index_builder import retriever


# Define a Pydantic model for hallucination grading.
class HallucinationGrader(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Indicates if the LLM generation is grounded in the facts: 'yes' means grounded, 'no' otherwise."
    )


structured_llm_hallucination = json_llm.with_structured_output(HallucinationGrader)


system = (
    "You are a grader assessing whether an LLM generation is grounded in or supported by a set of retrieved facts. "
    "Give a binary score 'yes' or 'no', where 'yes' means that the answer is grounded in the facts. "
    "Return your answer strictly as a JSON object with a single key 'binary_score'. "
    "For example, if the answer is grounded, output: binary_score: yes. "
    "Do not include any additional text or explanation."
)

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts:\n\n {documents}\n\n LLM generation:\n {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_hallucination

# Run
# question = "Agent Memory"
# docs = retriever.invoke(question)
# context = format_docs(docs)
# generated_answer = rag_chain.invoke({"context": context, "question": question})
# result = hallucination_grader.invoke({"documents": context, "generation": generated_answer})
# print("Grading result:", result.binary_score)

