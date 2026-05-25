from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

# The tools list - same as before
tools = [
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

# Run the same question 5 times and watch what changes
#user_question = "I'm planning a party for 23 people, each will have 3 drinks. Drinks cost $7 each. What's my budget?"
user_question = "Write me a short haiku about debugging."

for run_num in range(1, 6):
    print(f"\n--- Run {run_num} ---")
    messages = [{"role": "user", "content": user_question}]
    
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    for block in response.content:
        if block.type == "tool_use":
            print(f"  Tool: {block.name}({block.input})")
        elif block.type == "text":
            print(f"  Text: {block.text[:100]}...")