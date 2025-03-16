from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from services.llm_instance import json_llm

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

hallucination_grader = hallucination_prompt | json_llm | JsonOutputParser()

# Run
# question = "Agent Memory"
# docs = retriever.invoke(question)
# context = format_docs(docs)
# generated_answer = rag_chain.invoke({"context": context, "question": question})
# result = hallucination_grader.invoke({"documents": context, "generation": generated_answer})
# print("Grading result:", result)

