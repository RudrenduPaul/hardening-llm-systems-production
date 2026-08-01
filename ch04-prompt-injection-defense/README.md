# Chapter 4: Prompt Injection — Defense-in-Depth When the Model Cannot Refuse

Build a multi-layer defense-in-depth architecture against prompt injection — covering input scanning, role separation, tool validation, output filtering, and a privilege-scoped LLM client that limits what any single prompt can authorize.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch04_notebook.ipynb`](ch04_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch04-prompt-injection-defense/ch04_notebook.ipynb) | Interactive notebook: test each defense layer individually and as a full pipeline |
| [`ch04_scripts.py`](ch04_scripts.py) | `InjectionScanner`, `ToolCallValidator`, `OutputFilter`, `PrivilegeScopedLLMClient` |

## What this chapter builds

- **InjectionScanner** — regex + heuristic scanner for direct and indirect injection patterns
- **ToolCallValidator** — Pydantic schema enforcement for tool arguments before dispatch
- **OutputFilter** — post-generation scan for instruction-following leakage and data exfiltration patterns
- **PrivilegeScopedLLMClient** — wraps any LLM client; enforces least-privilege tool grants per request context
- **Defense pipeline** — composable chain: scan → validate scope → call LLM → filter output
- **pytest gate** — auto-generated test file covering blocked and allowed scenarios

## Prerequisites

```bash
pip install pydantic>=2.0
```

> **No API key required** — all defense components validate structure and patterns; no LLM calls needed.

## Key concepts

| Concept | Description |
|---------|-------------|
| Defense-in-depth | Multiple independent layers; one bypass does not compromise the full chain |
| Privilege scope | Each request carries a capability token; tool grants are bounded by that token |
| Indirect injection | Attacker-controlled content in retrieved documents that redirects model behavior |

## Running locally

```bash
cd ch04-prompt-injection-defense
jupyter notebook ch04_notebook.ipynb
```
