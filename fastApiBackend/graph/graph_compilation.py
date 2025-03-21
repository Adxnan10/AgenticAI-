from langgraph.graph import END, StateGraph, START
from graph.graph_definition import GraphState
# import nodes from graph flow
from graph.graph_flow import web_search, retrieve, grade_documents, generate, transform_query
# import edges from graph flow
from graph.graph_flow import route_question, decide_to_generate, grade_generation_v_documents_and_question

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    # TODO, This is stupid now all paths will end the flow, but I am doing it temporarily
    {
        "not supported": END,
        "useful": END,
        "not useful": END,
    },
)

# Compile
app = workflow.compile()
