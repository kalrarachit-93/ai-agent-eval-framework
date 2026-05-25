\# AI Agent Evaluation Framework



A small, focused framework for evaluating non-deterministic AI agents using property-based assertions and statistical run sampling.



Built while transitioning from 8 years in manual QA into AI engineering. Designed to capture the testing patterns that traditional QA practice misses when applied to LLM-based agents.



\## Why this exists



Traditional QA assertions break for AI systems. Consider:



```python

\# Traditional QA - this works for deterministic code

assert response == "The answer is 110,983."



\# AI QA - this fails 4 out of 5 times even when the agent is correct

\# because Claude rephrases: "110,983 is the result", "Total: 110,983", etc.

```



LLM outputs vary across runs even with identical inputs. Effective evaluation requires three shifts in approach:



1\. \*\*Property-based assertions\*\* instead of exact string matching

2\. \*\*Statistical run sampling\*\* to surface flakiness (pass/fail/flaky as three states, not two)

3\. \*\*Encoding multiple acceptable behaviors\*\* for ambiguous inputs



This framework demonstrates all three.



\## What it tests



A small calculator agent built on Claude, with 9 test cases covering:



\- Simple and multi-step arithmetic

\- Word problems that require interpretation

\- Negative tests (questions where the tool should NOT be used)

\- Order-of-operations edge cases

\- Ambiguous inputs that have multiple valid behaviors

\- Decimal precision handling



\## Architecture



Four parts, mirroring how production AI eval systems are structured:



| Part | What it does |

|------|--------------|

| \*\*Agent\*\* | Runs the full tool-use loop: model call → tool execution → tool result → continue until done |

| \*\*Evaluators\*\* | Small functions checking one property each: `tool\_was\_called`, `text\_contains`, `tool\_input\_contains` |

| \*\*Test cases\*\* | Each pairs an input with a list of property checks |

| \*\*Runner\*\* | Loops each test N times, aggregates pass rates, classifies as PASS / FLAKY / FAIL |



\## Key insight: the third state



Traditional QA has two outcomes: pass or fail. AI QA has three:



\- \*\*PASS\*\* — assertion held on every run (deterministic-ish behavior)

\- \*\*FLAKY\*\* — assertion held on some runs but not others (variance in this dimension)

\- \*\*FAIL\*\* — assertion failed on every run (consistent wrong behavior, or wrong test)



The flaky state is the most informative. It tells you \*where\* the variance lives and lets you decide what threshold is acceptable per-property.



\## Sample output

======================================================================
Running 9 test cases x 5 runs each
Test: Simple multiplication
Input: What's 1247 multiplied by 89?
[PASS] Called calculate tool: 5/5 (100%)
[PASS] Calculation involves 1247: 5/5 (100%)
[PASS] Final answer contains 110983: 5/5 (100%)
Test: Ambiguous arithmetic (acceptable: clarify OR best-guess)
Input: What do you get when you have 17 and 4?
[PASS] Either asked for clarification OR used the tool: 5/5 (100%)
[PASS] Did NOT hallucinate an answer without tool use: 5/5 (100%)
...
======================================================================
SUMMARY
Total assertions: 22
Fully passing:  22
Flaky:          0
Failing:        0

## A real iteration example

During development, the ambiguous arithmetic test initially asserted that the agent should always call the `calculate` tool. It failed 0/5 — Claude consistently refused to guess at the operation and asked the user for clarification instead.

The bug was in the test, not the agent. The refined test now accepts either valid behavior: tool use OR clarification language. This is the daily reality of AI evaluation work — failing tests often indicate flawed expectations rather than agent bugs.

## Setup

```bash
# Clone
git clone https://github.com/[your-username]/ai-agent-eval-framework
cd ai-agent-eval-framework

# Install
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Configure
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Run
python evals.py
```

## What I'd build next

- LLM-as-judge evaluator for free-form text outputs (haiku, summaries, etc.) where property checks don't apply
- Cost tracking per eval run (tokens × pricing)
- Regression dashboard: save results over time, surface when an assertion's pass rate drops
- Adversarial test case generation: use Claude to suggest edge cases for the agent under test
- Cross-model comparison: same eval suite against Claude Haiku, Opus, and other providers

## About me

Rachit Kalra — 8 years in manual QA, transitioning into AI engineering. Currently exploring agent evaluation, tool-use design, and the gap between traditional testing and LLM systems.

- LinkedIn: https://www.linkedin.com/in/rachit-kalra-softwaretestingengineer/
- Email: rachitkalra93@yahoo.com

