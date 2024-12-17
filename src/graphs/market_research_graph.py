from langgraph.graph import StateGraph, MessagesState, START
from src.agents.market_research_team.company_expert import company_expert_node
from src.agents.market_research_team.competitor_expert import competitor_expert_node
from src.agents.market_research_team.theoretical_market_expert import theoretical_market_expert_node
from src.agents.market_research_team.product_expert import product_expert_node
from src.agents.market_research_team.country_expert import country_expert_node
from src.agents.market_research_team.market_research_node import market_research_supervisor_node

research_builder = StateGraph(MessagesState)
research_builder.add_node("supervisor", market_research_supervisor_node)
research_builder.add_node("company_expert", company_expert_node)
research_builder.add_node("competitor_expert", competitor_expert_node)
research_builder.add_node("theoretical_market_expert", theoretical_market_expert_node)
research_builder.add_node("product_expert", product_expert_node)
research_builder.add_node("country_expert", country_expert_node)

research_builder.add_edge(START, "supervisor")
market_research_graph = research_builder.compile()

# Save diagram
with open("diagrams/market_research_graph.png", "wb") as f:
    f.write(market_research_graph.get_graph().draw_mermaid_png())
