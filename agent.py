from anthropic import Anthropic
from dotenv import load_dotenv
import datetime

load_dotenv()
client = Anthropic()

# ---- Step 1: Define the tools (these are just Python functions) ----

def get_current_time(timezone: str = "UTC") -> str:
    """Returns the current time. Pretends to handle timezone for demo purposes."""
    now = datetime.datetime.now()
    return f"The current time is {now.strftime('%Y-%m-%d %H:%M:%S')} ({timezone})"

def calculate(expression: str) -> str:
    """Safely evaluates a math expression like '23 * 47 + 100'."""
    try:
        # Only allow math characters - basic safety
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: only numbers and basic math operators allowed"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# ---- Step 2: Describe the tools to Claude in its required format ----

tools = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time. Use this when the user asks about time, date, or 'now'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name, e.g. 'UTC' or 'IST'. Defaults to UTC."
                }
            },
            "required": []
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use this for any math the user asks about.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression like '15 * 23' or '(100 + 50) / 2'"
                }
            },
            "required": ["expression"]
        }
    }
]

# ---- Step 3: A dispatcher that maps tool names to actual functions ----

def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_current_time":
        return get_current_time(**tool_input)
    elif tool_name == "calculate":
        return calculate(**tool_input)
    else:
        return f"Unknown tool: {tool_name}"

# ---- Step 4: Ask Claude something that requires a tool ----

user_question = "What's 1247 multiplied by 89, and what time is it right now?"

print(f"User: {user_question}\n")

messages = [{"role": "user", "content": user_question}]

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    tools=tools,
    messages=messages
)

print(f"Claude's first response (stop_reason: {response.stop_reason}):")
for block in response.content:
    if block.type == "text":
        print(f"  Text: {block.text}")
    elif block.type == "tool_use":
        print(f"  Tool call: {block.name}({block.input})")

# ---- Step 5: If Claude wants to use tools, run them and send results back ----

while response.stop_reason == "tool_use":
    # Add Claude's response to the message history
    messages.append({"role": "assistant", "content": response.content})
    
    # Run each tool Claude requested and collect the results
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = run_tool(block.name, block.input)
            print(f"\n  → Ran {block.name}: {result}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result
            })
    
    # Send the tool results back to Claude
    messages.append({"role": "user", "content": tool_results})
    
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

# ---- Step 6: Print Claude's final answer ----

print("\n" + "="*50)
print("Claude's final answer:")
for block in response.content:
    if block.type == "text":
        print(block.text)