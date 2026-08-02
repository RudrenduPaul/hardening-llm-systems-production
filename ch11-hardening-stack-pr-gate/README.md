# Chapter 11: The Hardening Stack — The PR-Gate That Blocks Unsafe Deploys

Assemble the six hardening layers built across Chapters 2–10 into a single PR-gate orchestrator: input guardrails, LLM gateway, observability and agent monitoring, evaluation gate, harmful-output and bias gates, and the PR-gate itself. Every check from the preceding chapters arrives as a single merge-blocking signal.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch11_notebook.ipynb`](ch11_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch11-hardening-stack-pr-gate/ch11_notebook.ipynb) | Interactive notebook: run each layer of the hardening stack against a demo system, then the full ten-check PR-gate |
| [`ch11_scripts.py`](ch11_scripts.py) | `PRHardeningGate`, `CIHardeningOrchestrator`, `StackLatencyProfiler`, `StackLatencyBudget`, `RoutingPolicy`, reference-stack builders, `generate_migration_plan` |

## What this chapter builds

- **`get_nemo_guardrails_app()`** (Listing 11.1b) — FastAPI app wrapping NeMo Guardrails with the Colang config for jailbreak, off-topic, and PII-leakage rails
- **`build_guardrails_ai_rag_pipeline()` / `run_guardrails_ai_rag()`** (Listing 11.2) — Guardrails AI validator pipeline that enforces a JSON schema and re-asks on toxic or malformed RAG output
- **`write_litellm_proxy_config()` / `get_litellm_client()` / `litellm_completion()`** (Listing 11.3) — LiteLLM proxy config and client with latency-based routing and fallback across three models
- **`PresidioMiddleware`** — ASGI middleware that scrubs PII from request and response bodies using Microsoft Presidio before they reach the LLM or the client
- **`setup_observability()` / `trace_llm_call()`** (Listing 11.4) — OpenTelemetry span + Langfuse generation wiring for every LLM call
- **`CIHardeningOrchestrator`** — runs the deepeval suite and a Garak adversarial probe run, writes a combined `ci-hardening-report.json`
- **`StackLatencyProfiler`** (Listing 11.5) — per-layer P50/P99 latency measurement across the six request-path layers
- **`StackLatencyBudget`** (Listing 11.5b) — allocates a total P99 budget across layers and reports `ok` / `warning` / `over_budget` status per layer (section 11.6.1)
- **`RoutingPolicy` / `DegradationMode`** — health- and cost-aware endpoint selection with automatic fallback and refusal modes
- **`PRHardeningGate`** (Listing 11.9) — the ten merge-blocking checks from section 11.8; `run()` prints a pass/fail table and calls `sys.exit(1)` on any failure (or raises `RuntimeError` when `strict=False`)
- **`build_hardened_chat_assistant()` / `build_hardened_rag_app()` / `build_hardened_agent()`** (Listings 11.6–11.8) — the three reference stacks: LangChain + NeMo Guardrails, LlamaIndex + Pinecone + Guardrails AI, LangGraph + MCP with tool allowlist and human approval gate
- **`generate_migration_plan()` / `print_migration_plan()`** (Listing 11.8b) — sequenced, rollback-annotated migration steps for adding the stack to an existing LangChain, LlamaIndex, or LangGraph deployment

## Prerequisites

```bash
pip install pyyaml
# Optional, per layer (graceful fallback via stub results if missing):
pip install nemoguardrails==0.9.0 guardrails-ai==0.5.0 litellm==1.35.0 \
            opentelemetry-sdk==1.21.0 opentelemetry-exporter-otlp==1.21.0 \
            langfuse==2.28.0 deepeval==0.21.7 garak==0.10.0 \
            presidio-analyzer==2.2.354 presidio-anonymizer==2.2.354 \
            langchain==0.3.0 langchain-openai==0.2.0 \
            llama-index==0.11.0 pinecone-client==4.1.0 \
            langgraph==0.2.0 fastapi>=0.110.0
```

> **No API key required for the demo** — `python3 ch11_scripts.py` runs end-to-end against stub/synthetic data when an optional dependency (deepeval, Garak, Presidio, and so on) isn't installed. Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `LANGFUSE_*` keys for live execution against a real model and Langfuse project.

## Key concepts

| Concept | Description |
|---------|-------------|
| PR-gate | `PRHardeningGate` aggregates pre-computed report dicts from the checks each earlier chapter's companion code produces (ch03 shadow-traffic/canary, ch05 tenant isolation and retrieval anomaly, ch07 agent scope, ch08 PII and right-to-erasure, ch09 content safety, ch10 Annex IV) into one merge-blocking signal — it does not re-run those evaluations itself |
| Ten checks | Numbered as in section 11.8; `PRHardeningGate.run()` executes them in cost order (Annex IV first, adversarial and red-team scans last) regardless of the numbering |
| Deployment type | `PRHardeningGate(deployment_type="chat" | "rag" | "agent")` — the retrieval-grounding, RAG-tenant-isolation, and agent-scope-containment checks skip (pass, non-blocking) when they don't apply to the deployment type |

## Running locally

```bash
cd ch11-hardening-stack-pr-gate
python3 ch11_scripts.py          # end-to-end demo of every component, including the ten-check gate
jupyter notebook ch11_notebook.ipynb
```
