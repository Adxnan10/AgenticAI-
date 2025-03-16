"""
query_router module sets up a query router to determine the appropriate datasource
for user questions. It routes queries to either a vectorstore (for specific topics) or web search, for now.
"""
from pydantic import BaseModel, Field
from typing import Literal
from services.llm_instance import json_llm
from langchain_core.prompts import PromptTemplate


# Define the Pydantic model for routing
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ..., description="Given a user question, choose to route it to 'web_search' or 'vectorstore'."
    )

# Create the structured output LLM wrapper using the Pydantic model
structured_llm_router = json_llm.with_structured_output(RouteQuery)

prompt = PromptTemplate(
    template=(
        "system You are an expert at routing a user question to a vectorstore or web_search. "
        "Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks. "
        "Otherwise, use web_search. Return a JSON object with a single key \"datasource\" whose value must be "
        "either \"web_search\" or \"vectorstore\". Do not include any extra text or explanation.\n"
        "Question to route: {question}\n"
        "assistant"
    ),
    input_variables=["question"],
)

question_router = prompt | structured_llm_router


