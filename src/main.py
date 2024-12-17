import argparse
from src.graphs.super_graph import super_graph
from src.graphs.research_graph import research_graph
from src.graphs.paper_writing_graph import paper_writing_graph
from src.graphs.market_research_graph import market_research_graph

import os
import getpass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def set_env_var(var_name: str):
    if not os.environ.get(var_name):
        os.environ[var_name] = getpass.getpass(f"Please provide your {var_name}: ")

# Load keys if not set
set_env_var("OPENAI_API_KEY")
set_env_var("TAVILY_API_KEY")


def run_graph(graph_name: str, user_input: str):
    if graph_name == "super":
        graph = super_graph
    elif graph_name == "market":
        graph = market_research_graph
    elif graph_name == "research":
        graph = research_graph
    elif graph_name == "writing":
        graph = paper_writing_graph
    else:
        raise ValueError(f"Unknown graph: {graph_name}")

    for state in graph.stream({
        "messages": [
            (
                "user",
                user_input
            )
        ]
    },
    {"recursion_limit": 100},):
        print(state)
        print("---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a hierarchical agent team graph.")
    parser.add_argument("graph", choices=["super", "research", "writing", "market"], help="Which graph to run.")
    parser.add_argument("query", help="User query or input to the agents.")
    args = parser.parse_args()

    run_graph(args.graph, args.query)