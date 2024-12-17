from langgraph.graph import StateGraph, MessagesState, START
from src.agents.doc_writing_agents import (
    doc_writing_supervisor_node,
    doc_writing_node,
    note_taking_node,
    chart_generating_node,
)

paper_writing_builder = StateGraph(MessagesState)
paper_writing_builder.add_node("supervisor", doc_writing_supervisor_node)
paper_writing_builder.add_node("doc_writer", doc_writing_node)
paper_writing_builder.add_node("note_taker", note_taking_node)
paper_writing_builder.add_node("chart_generator", chart_generating_node)

paper_writing_builder.add_edge(START, "supervisor")
paper_writing_graph = paper_writing_builder.compile()

# Save diagram
with open("diagrams/paper_writing_graph.png", "wb") as f:
    f.write(paper_writing_graph.get_graph().draw_mermaid_png())
