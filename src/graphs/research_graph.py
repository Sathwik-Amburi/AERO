from langgraph.graph import StateGraph, MessagesState, START
from src.agents.research_agents import research_supervisor_node, search_node, web_scraper_node

research_builder = StateGraph(MessagesState)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()

# Save diagram
with open("diagrams/research_graph.png", "wb") as f:
    f.write(research_graph.get_graph().draw_mermaid_png())
