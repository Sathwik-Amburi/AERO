from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from typing import Annotated

repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute."]
) -> str:
    """Execute Python code in a REPL environment and return stdout."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
