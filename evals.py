"""
A tiny eval framework for testing AI agents.
"""

from anthropic import Anthropic
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Callable
import time

load_dotenv()
client = Anthropic()

# ============================================================
# PART 1: Tools and the agent under test
# ============================================================

tools = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use this for any math the user asks about.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression like '15 * 23'"
                }
            },
            "required": ["expression"]
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Actually runs a tool and returns the result as a string."""
    if tool_name == "calculate":
        try:
            allowed = set("0123456789+-*/(). ")
            expr = tool_input["expression"]
            if not all(c in allowed for c in expr):
                return "Error: only numbers and basic math operators allowed"
            return str(eval(expr))
        except Exception as e:
            return f"Error: {e}"
    return f"Unknown tool: {tool_name}"


def run_agent(user_message: str) -> dict:
    """
    Runs the full agent loop: initial call, runs any tools, sends results back,
    continues until the agent stops. Returns what happened across the whole run.
    """
    messages = [{"role": "user", "content": user_message}]
    all_tool_calls = []
    final_text = ""
    last_stop_reason = None

    for iteration in range(10):  # safety cap
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        last_stop_reason = response.stop_reason

        tool_uses_this_turn = []
        for block in response.content:
            if block.type == "tool_use":
                all_tool_calls.append({"name": block.name, "input": block.input})
                tool_uses_this_turn.append(block)
            elif block.type == "text":
                final_text = block.text

        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_block in tool_uses_this_turn:
            result = execute_tool(tool_block.name, tool_block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": str(result)
            })
        messages.append({"role": "user", "content": tool_results})

    return {
        "tool_calls": all_tool_calls,
        "text": final_text,
        "stop_reason": last_stop_reason,
    }


# ============================================================
# PART 2: Evaluators
# ============================================================

def tool_was_called(result: dict, tool_name: str) -> bool:
    return any(tc["name"] == tool_name for tc in result["tool_calls"])


def tool_input_contains(result: dict, tool_name: str, substring: str) -> bool:
    for tc in result["tool_calls"]:
        if tc["name"] == tool_name:
            if substring in str(tc["input"]):
                return True
    return False


def text_contains(result: dict, substring: str) -> bool:
    return substring.lower() in result["text"].lower()


# ============================================================
# PART 3: Test cases
# ============================================================

@dataclass
class TestCase:
    name: str
    user_input: str
    checks: list


test_cases = [
    TestCase(
    name="Ambiguous arithmetic (acceptable: clarify OR best-guess)",
    user_input="What do you get when you have 17 and 4?",
    checks=[
        # Good behavior #1: Asks for clarification
        # Good behavior #2: Makes a reasonable assumption and calculates
        # Bad behavior: Hallucinates a random number without using the tool
        (
            "Either asked for clarification OR used the tool",
            lambda r: (
                tool_was_called(r, "calculate")
                or text_contains(r, "clarif")
                or text_contains(r, "what operation")
                or text_contains(r, "what would you like")
                or text_contains(r, "do you mean")
                or text_contains(r, "could you specify")
            )
        ),
        (
            "Did NOT hallucinate an answer without tool use",
            # If Claude didn't use the tool, the response shouldn't contain confident numbers
            # This is harder to check perfectly, but we can check for clarifying language
            lambda r: tool_was_called(r, "calculate") or any(
                phrase in r["text"].lower()
                for phrase in ["clarif", "specify", "mean", "could you", "would you", "operation"]
            )
        ),
    ]
),
    TestCase(
        name="Simple multiplication",
        user_input="What's 1247 multiplied by 89?",
        checks=[
            ("Called calculate tool", lambda r: tool_was_called(r, "calculate")),
            ("Calculation involves 1247", lambda r: tool_input_contains(r, "calculate", "1247")),
            ("Calculation involves 89", lambda r: tool_input_contains(r, "calculate", "89")),
            ("Final answer contains 110983", lambda r: text_contains(r, "110983") or text_contains(r, "110,983")),
        ]
    ),
    TestCase(
        name="Word problem with multiplication",
        user_input="I have 15 boxes with 23 items each. How many items total?",
        checks=[
            ("Called calculate tool", lambda r: tool_was_called(r, "calculate")),
            ("Calculation involves 15", lambda r: tool_input_contains(r, "calculate", "15")),
            ("Calculation involves 23", lambda r: tool_input_contains(r, "calculate", "23")),
            ("Final answer contains 345", lambda r: text_contains(r, "345")),
        ]
    ),
    TestCase(
        name="Non-math question (should NOT use tool)",
        user_input="What's the capital of France?",
        checks=[
            ("Did NOT call calculate tool", lambda r: not tool_was_called(r, "calculate")),
            ("Mentions Paris", lambda r: text_contains(r, "Paris")),
        ]
    ),
    TestCase(
        name="Multi-step word problem",
        user_input="I'm buying 5 shirts at $24 each plus 3 pairs of pants at $40 each. What's my total?",
        checks=[
            ("Called calculate tool", lambda r: tool_was_called(r, "calculate")),
            ("Final answer mentions 240", lambda r: text_contains(r, "240")),
        ]
    ),
TestCase(
    name="Math trick question (order of operations)",
    user_input="What is 6 plus 4 times 3?",
    checks=[
        # Correct answer is 18 (multiplication first), NOT 30
        ("Called calculate tool", lambda r: tool_was_called(r, "calculate")),
        ("Final answer contains 18", lambda r: text_contains(r, "18")),
        ("Final answer does NOT contain 30", lambda r: not text_contains(r, "30")),
    ]
),
TestCase(
    name="Multi-step with subtraction",
    user_input="I started with $500. I bought 3 books at $45 each and a coffee for $6.50. How much do I have left?",
    checks=[
        ("Called calculate tool", lambda r: tool_was_called(r, "calculate")),
        ("Final answer mentions 358.50 or 358.5", lambda r: text_contains(r, "358.5")),
    ]
),
TestCase(
    name="Tool misuse - greeting should not trigger tool",
    user_input="Hi! How are you doing today?",
    checks=[
        ("Did NOT call calculate tool", lambda r: not tool_was_called(r, "calculate")),
    ]
),
TestCase(
    name="Decimal precision",
    user_input="What is 1 divided by 3?",
    checks=[
        ("Called calculate tool", lambda r: tool_was_called(r, "calculate")),
        # 1/3 is 0.333... - check the agent handles the result reasonably
        ("Final answer mentions 0.33 or one-third", 
         lambda r: text_contains(r, "0.33") or text_contains(r, "0.333") or text_contains(r, "third")),
    ]
),
]


# ============================================================
# PART 4: The runner
# ============================================================

RUNS_PER_TEST = 5

def run_evals():
    print("=" * 70)
    print(f"Running {len(test_cases)} test cases x {RUNS_PER_TEST} runs each")
    print("=" * 70)

    overall_results = []

    for tc in test_cases:
        print(f"\nTest: {tc.name}")
        print(f"   Input: {tc.user_input}")

        check_pass_counts = {desc: 0 for desc, _ in tc.checks}

        for run_i in range(RUNS_PER_TEST):
            try:
                result = run_agent(tc.user_input)
                for desc, check_fn in tc.checks:
                    if check_fn(result):
                        check_pass_counts[desc] += 1
            except Exception as e:
                print(f"   Run {run_i+1} errored: {e}")
            time.sleep(0.5)

        for desc, _ in tc.checks:
            passed = check_pass_counts[desc]
            rate = passed / RUNS_PER_TEST * 100
            if passed == RUNS_PER_TEST:
                status = "PASS"
            elif passed > 0:
                status = "FLAKY"
            else:
                status = "FAIL"
            print(f"   [{status}] {desc}: {passed}/{RUNS_PER_TEST} ({rate:.0f}%)")
            overall_results.append((tc.name, desc, passed, RUNS_PER_TEST))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(overall_results)
    fully = sum(1 for _, _, p, t in overall_results if p == t)
    flaky = sum(1 for _, _, p, t in overall_results if 0 < p < t)
    failing = sum(1 for _, _, p, t in overall_results if p == 0)

    print(f"Total assertions: {total}")
    print(f"  Fully passing:  {fully}")
    print(f"  Flaky:          {flaky}")
    print(f"  Failing:        {failing}")


if __name__ == "__main__":
    run_evals()