from typing import Literal, TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import  MessagesState,END
from langgraph.types import Command


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
f"""
You are a supervisor tasked with managing a conversation between the following workers: {members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. 

Instructions for Task Management:  
1. Start by identifying the worker most relevant to the user’s request and assign them their task.  
2. Once a worker completes their task, collect their results and decide the next worker to act, ensuring logical flow and addressing task dependencies.  
3. Monitor the progress of all workers and confirm all necessary tasks are completed.  
4. After gathering results from all workers, pass all collected data to the **Theoretical Market Expert** for final synthesis and analysis.  
5. When the Theoretical Market Expert provides the final output, respond with **FINISH**.  

Workflow Example:  
1. Assign a task to the first relevant worker.  
2. After their response, delegate tasks to the next logical worker based on dependencies.  
3. Once all data is collected, instruct the Theoretical Market Expert to generate the final report.  
4. Respond with **FINISH** after receiving the final report.  

Your role is to efficiently orchestrate the workflow between workers and ensure a cohesive, complete, and actionable output for the user.
"""
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto)

    return supervisor_node