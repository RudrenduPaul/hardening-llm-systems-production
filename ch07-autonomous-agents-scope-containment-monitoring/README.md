# Chapter 7: Autonomous Agents — Scope, Containment, and Monitoring

Build the containment layer for autonomous LLM agents — enforcing MCP tool allowlists, validating trust levels across multi-agent message boundaries, scoping credentials per task, and sandboxing subprocess execution so a compromised tool cannot escalate privileges.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch07_notebook.ipynb`](ch07_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch07-autonomous-agents-scope-containment-monitoring/ch07_notebook.ipynb) | Interactive notebook: test each containment component with synthetic tool definitions and messages |
| [`ch07_scripts.py`](ch07_scripts.py) | `MCPToolAllowlistEnforcer`, `TrustLevelWrapper`, `ScopedCredentialManager`, `SandboxedSubprocessExecutor` |

## What this chapter builds

- **MCPToolAllowlistEnforcer** — SHA-256 hash-pinned tool registry; blocks any tool not in the approved set (Listing 7.1)
- **MCPToolDescriptionValidator / DescriptionValidationResult** — regex injection scanner on tool descriptions; prevents poisoned tool metadata (Listing 7.2)
- **TrustLevelWrapper** — `TrustLevel` IntEnum (SYSTEM / AGENT / EXTERNAL) enforced at every message boundary; `downgrade_trust_at_boundary()` forces EXTERNAL trust on any message crossing an external boundary (Listing 7.3)
- **ScopedCredentialManager** — AWS STS-style per-scope temporary credentials with TTL enforcement (Listing 7.4)
- **ActionCategorizer + ConfirmationGate** — regex-based risk-tier classification (READ_ONLY / REVERSIBLE / IRREVERSIBLE / DESTRUCTIVE) with an async human-in-the-loop gate for high-risk actions (Listing 7.5)
- **SandboxedSubprocessExecutor** — resource-limited subprocess execution with allowlist enforcement (Listing 7.6)
- **ApprovalRequest / AgentApprovalQueue** — structured approval request (agent goal, plain-English action, predicted outcome, fallback plan) with an async fail-closed approval queue (Listing 7.7)
- **pytest test suite** — full CI gate for all containment components
- **AgentMemoryValidator** — detects memory segment drift using source-weighted cosine similarity; classifies injection attempts across external content, tool outputs, and user messages (Listing 7.12)
- **AgentComplexityScorer** — implements CSA CDR framework; classifies cognitive degradation severity at 1.75x/2.5x/4.0x baseline multipliers; triggers rebrief or restart (Listing 7.13)
- **TrifectaScore** — scores an agent architecture against Simon Willison's lethal trifecta (private data access, untrusted content exposure, exfiltration capability) (Listing 7.0)
- **trace_agent_step** — decorator for tracing individual LLM calls with OpenTelemetry spans; graceful no-op when OTel SDK is absent (Listing 7.8)
- **InstrumentedAgent** — agent wrapper that emits OTel spans for planning and tool-execution steps (Listing 7.9)
- **AgentTripwireDetector / TripwireEvent** — three named tripwire rules: UNAUTHORIZED_TOOL (P0), EXCESSIVE_READ (P1), WRITE_WITHOUT_READ (P2) (Listing 7.10)
- **CUSUMActionRateMonitor** — bidirectional CUSUM controller for detecting agentic action rate anomalies (Listing 7.11)
- **ScopeTestCase / test_agent_scope_and_telemetry** — unified CI gate verifying scope enforcement and telemetry instrumentation as release-blocking signals (Listing 7.14)

Supplementary components below are referenced in chapter prose but are not printed as numbered chapter listings:

- **SignedAgentMessage** — signed agent message with HMAC-SHA256 origin verification; checks freshness, scope, and signature at every trust boundary (implements the design described in section 7.5.1)
- **TracedAgentSession** — Langfuse session tracing for full prompt/response/goal capture; graceful no-op when env vars missing (implements the design described in section 7.9.2)
- **calibrate_tripwire_threshold** — empirical FPR-based threshold calibration from an action-history log (implements the calibration approach described in sections 7.10.1-7.10.2)

## Prerequisites

```bash
pip install langgraph==0.2.0 langchain-openai==0.2.0
```

> **No API key required** — all demonstrations use synthetic tool definitions and messages; no agent actually executes.

## Key concepts

| Concept | Description |
|---------|-------------|
| Hash-pinned allowlist | Tools are approved by content hash, not by name; renamed or updated tools require re-approval |
| Trust level boundary | Messages crossing trust levels (e.g., EXTERNAL → AGENT) are sanitized and logged; trust can only be downgraded, never upgraded, as a message moves outward |
| Scope-bound credentials | Each tool invocation receives credentials scoped only to that tool's declared permissions |

## Running locally

```bash
cd ch07-autonomous-agents-scope-containment-monitoring
jupyter notebook ch07_notebook.ipynb
```
