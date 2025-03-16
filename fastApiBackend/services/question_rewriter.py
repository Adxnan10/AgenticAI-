from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from services.llm_instance import json_llm

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n for 
vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. \n reutrn 
a json with two keys only, improved question and reason. Improved question key must be only the new rewritten 
question without any additional text or explanation. \n"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
question_rewriter = re_write_prompt | json_llm | JsonOutputParser()
# run
# question = "agent memory"
# print(question_rewriter.invoke({"question": question}))
