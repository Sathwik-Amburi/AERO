from src.agents.market_research_team.utils import make_supervisor_node
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini")
market_research_supervisor_node = make_supervisor_node(llm, ["company_expert", "competitor_expert", "theoretical_market_expert","product_expert","country_expert"])