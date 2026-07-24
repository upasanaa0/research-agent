from langchain_core.messages import SystemMessage, ToolMessage
from llm import llm
from tools import TOOLS, TOOL_MAP

llm_with_tools = llm.bind_tools(TOOLS)

MAX_TOOL_RESULT_CHARS = 3000  # truncate each tool result to this length


def execute_tool_calls(response_message) -> list:
    tool_messages = []
    if not hasattr(response_message, "tool_calls") or not response_message.tool_calls:
        return tool_messages

    for tool_call in response_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"\n[tool_call] {tool_name}({tool_args})")

        if tool_name in TOOL_MAP:
            result = TOOL_MAP[tool_name].invoke(tool_args)
        else:
            result = f"Tool '{tool_name}' not found."

        # Truncate to avoid context overflow
        result_str = str(result)
        if len(result_str) > MAX_TOOL_RESULT_CHARS:
            result_str = result_str[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"

        print(f"[tool_result] {result_str[:200]}...")

        tool_messages.append(
            ToolMessage(content=result_str, tool_call_id=tool_call_id)
        )

    return tool_messages


def invoke_with_tools(system_prompt: str, messages: list) -> str:
    current_messages = [SystemMessage(content=system_prompt)] + list(messages)

    while True:
        response = llm_with_tools.invoke(current_messages)
        tool_messages = execute_tool_calls(response)
        if not tool_messages:
            return response.content
        current_messages.append(response)
        current_messages.extend(tool_messages)