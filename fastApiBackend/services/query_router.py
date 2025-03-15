"""
query_router module sets up a query router to determine the appropriate datasource
for user questions. It routes queries to either a vectorstore (for specific topics) or web search, for now.
"""
from langchain_core.prompts import PromptTemplate
from services.llm_instance import llm
from langchain_core.output_parsers import JsonOutputParser

# Prompt
prompt = PromptTemplate(
    template="""system You are an expert at routing a user question to a vectorstore or web search. Use the 
    vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks. You do not need to be 
    stringent with the keywords in the question related to these topics. Otherwise, use web-search. Give a binary 
    choice 'web_search' or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no preamble or explanation. Question to route: {question} assistant""",
    input_variables=["question"],
)
# with_structured_output does not work with Llama:
# structured_llm_router = llm.with_structured_output(RouteQuery)
# So I will be using a work around that should get the job done
question_router = prompt | llm | JsonOutputParser()

