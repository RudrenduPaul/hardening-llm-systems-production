# Chapter 7: Autonomous Agents — Scope, Containment, and Monitoring

Build the containment layer for autonomous LLM agents — enforcing MCP tool allowlists, validating trust levels across multi-agent message boundaries, scoping credentials per task, and sandboxing subprocess execution so a compromised tool cannot escalate privileges.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch07_notebook.ipynb`](ch07_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch07-autonomous-agents-scope-containment-monitoring/ch07_notebook.ipynb) | Interactive notebook: test each containment component with synthetic tool definitions and messages |
| [`ch07_scripts.py`](ch07_scripts.py) | `MCPToolAllowlistEnforcer`, `TrustLevelWrapper`, `ScopedCredentialManager`, `SandboxedExecutor` |

## What this chapter builds

- **MCPToolAllowlistEnforcer** — SHA-256 hash-pinned tool registry; blocks any tool not in the approved set
- **MCPToolDescriptionValidator** — regex injection scanner on tool descriptions; prevents poisoned tool metadata
- **TrustLevelWrapper** — `TrustLevel` enum (SYSTEM / OPERATOR / USER / EXTERNAL) enforced at every message boundary
- **ScopedCredentialManager** — AWS STS-style per-scope temporary credentials with TTL enforcement
- **ActionCategorizer + ApprovalGate** — async human-in-the-loop gate for high-risk actions
- **SandboxedSubprocessExecutor** — resource-limited subprocess execution with blocklist enforcement
- **pytest test suite** — full CI gate for all containment components
- **AgentMemoryValidator** — detects memory segment drift using source-weighted cosine similarity; classifies injection attempts across external content, tool outputs, and user messages
- **AgentComplexityScorer** — implements CSA CDR framework; classifies cognitive degradation severity at 1.75x/2.5x/4.0x baseline multipliers; triggers rebrief or restart
- **TrifectaScore** — scores an agent architecture against Simon Willison's lethal trifecta (private data access, untrusted content exposure, exfiltration capability)
- **SignedAgentMessage** — signed agent message with HMAC-SHA256 origin verification; checks freshness, scope, and signature at every trust boundary
- **trace_agent_step** — decorator for tracing individual LLM calls with OpenTelemetry spans; graceful no-op when OTel SDK is absent
- **InstrumentedAgent** — agent wrapper that emits OTel spans for planning and tool-execution steps
- **TracedAgentSession** — Langfuse session tracing for full prompt/response/goal capture; graceful no-op when env vars missing
- **calibrate_tripwire_threshold** — empirical FPR-based threshold calibration from an action-history log
- **AgentTripwireDetector / TripwireEvent** — three named tripwire rules: UNAUTHORIZED_TOOL (P0), EXCESSIVE_READ (P1), WRITE_WITHOUT_READ (P2)
- **CUSUMActionRateMonitor** — bidirectional CUSUM controller for detecting agentic action rate anomalies
- **ScopeTestCase / test_agent_scope_and_telemetry** — unified CI gate verifying scope enforcement and telemetry instrumentation as release-blocking signals

## Prerequisites

```bash
pip install langgraph==0.2.0 langchain-openai==0.2.0
```

> **No API key required** — all demonstrations use synthetic tool definitions and messages; no agent actually executes.

## Key concepts

| Concept | Description |
|---------|-------------|
| Hash-pinned allowlist | Tools are approved by content hash, not by name; renamed or updated tools require re-approval |
| Trust level boundary | Messages crossing trust levels (e.g., EXTERNAL → OPERATOR) are sanitized and logged |
| Scope-bound credentials | Each tool invocation receives credentials scoped only to that tool's declared permissions |

## Running locally

```bash
cd ch07-autonomous-agents-scope-containment-monitoring
jupyter notebook ch07_notebook.ipynb
```
