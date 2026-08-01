# Chapter 11: The Hardening Stack — The PR-Gate That Blocks Unsafe Deploys

Assemble the six hardening layers built across Chapters 2–10 into a single PR-gate orchestrator: input guardrails, LLM gateway, observability and agent monitoring, evaluation gate, harmful-output and bias gates, and the PR-gate itself. Every check from the preceding chapters arrives as a single merge-blocking signal.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch11_notebook.ipynb`](ch11_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch11-hardening-stack-pr-gate/ch11_notebook.ipynb) | Interactive notebook: run the full hardening stack against a demo system, generate gate report |
| [`ch11_scripts.py`](ch11_scripts.py) | `HardeningPRGate`, `GatePolicy`, `GateResult`, `HardeningStackOrchestrator` |

## What this chapter builds

- **HardeningPRGate** — orchestrates all six layer checks in a single sequential or parallel run
- **GatePolicy** — configurable pass/fail thresholds per check type; separate policies for pre-prod and prod branches
- **GateResult** — structured result with per-check status, score, threshold, and remediation pointer
- **HardeningStackOrchestrator** — async-capable runner with early-exit on critical failures
- **GitHub Actions workflow** — ready-to-use `.github/workflows/hardening-gate.yml` template
- **NeMo Guardrails integration** — conversation-level policy enforcement via config files (optional)
- **Guardrails AI integration** — structured output validation schema enforcement (optional)
- **Langfuse observability** — full gate run recorded as a Langfuse session for audit trail

## Prerequisites

```bash
pip install pydantic>=2.0 langchain-core
# Optional (graceful fallback if missing):
pip install nemoguardrails==0.9.0 guardrails-ai==0.5.0 langfuse==2.28.0
```

> **No API key required for demo** — the gate runs against a simulated system with pre-seeded check results. Set `OPENAI_API_KEY` and `LANGFUSE_*` keys for live execution.

## Key concepts

| Concept | Description |
|---------|-------------|
| PR gate | Automated quality gate that blocks merges until all hardening checks pass |
| Check composition | Each check from Chapters 1–11 is a `GateCheck` with standard interface — plug any in or out |
| Early-exit policy | Critical failures (injection bypass, PII leakage above threshold) halt the gate immediately |

## Running locally

```bash
cd ch11-hardening-stack-pr-gate
jupyter notebook ch11_notebook.ipynb
```
