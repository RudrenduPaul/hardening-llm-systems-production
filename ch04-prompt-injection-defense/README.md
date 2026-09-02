# Chapter 4: Prompt Injection Defense-in-Depth When the Model Cannot Refuse

Build a multi-layer defense-in-depth architecture against prompt injection, covering input scanning, role separation, tool validation, output filtering, and a privilege-scoped LLM client that limits what any single prompt can authorize.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch04_notebook.ipynb`](ch04_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch04-prompt-injection-defense/ch04_notebook.ipynb) | Interactive notebook: test each defense layer individually and as a full pipeline |
| [`ch04_scripts.py`](ch04_scripts.py) | `MCPToolDefinition`, `embedding_detector`, `PermissionSet`, `ScopeToken`, `PrivilegeScopedLLMClient`, `OutputExfiltrationFilter`, `BlastRadiusLimiter`, `CapabilityToken`, `CapabilityRuntime`, `PromptInjectionDetector`, `InjectionDefensePipeline` |
| [`ch04_injection_tests.py`](ch04_injection_tests.py) | pytest regression suite covering the validator, scope token, exfiltration filter, detector, and full pipeline |
| [`ch04_prompt_hash_gate.py`](ch04_prompt_hash_gate.py) | Listing 4.7: standalone SHA-256 prompt-hash gate for CI |

## What this chapter builds

- **`MCPToolDefinition`**: Pydantic schema validator for MCP tool definitions; rejects injection-like patterns in tool descriptions and enforces a safe naming policy (listing 4.1)
- **`embedding_detector`**: optional semantic-similarity injection check (sentence-transformers + scikit-learn) that catches paraphrased attacks a keyword filter misses (section 4.5.2)
- **`PermissionSet`**: statically declared, verified permission set (`allowed_reads`/`allowed_writes`/`allowed_network_calls`) with `CUSTOMER_SUPPORT_PERMISSIONS` and `FINANCIAL_REPORTING_PERMISSIONS` demonstration instances (section 4.6.1)
- **`ScopeToken` + `PrivilegeScopedLLMClient`**: OAuth-style scope tokens; strips any tool whose required scope isn't granted before the request reaches the model (listing 4.2)
- **`OutputExfiltrationFilter`**: post-generation scan for URLs, base64 blobs, credentials, and PII in model output (listing 4.3)
- **`BlastRadiusLimiter`**: rate limiting and confirmation gates for high-impact tool actions (listing 4.4)
- **`CapabilityToken` + `CapabilityRuntime`**: simplified CaMeL-inspired capability tokens issued and validated by a trusted runtime the model never sees (listing 4.5)
- **`PromptInjectionDetector`**: two-layer detection pipeline: regex heuristics plus an optional LLM Guard scanner (listing 4.6)
- **`InjectionDefensePipeline`**: composable chain: validate tools → check scope token → detect injection → apply blast-radius limits → filter output
- **pytest gate**: `ch04_injection_tests.py` covers blocked and allowed scenarios across every component above

## Prerequisites

```bash
pip install pydantic>=2.0
```

Optional, for the two components that degrade gracefully without them:

```bash
pip install llm-guard==0.3.12                          # PromptInjectionDetector layer 2
pip install sentence-transformers==2.6.0 scikit-learn   # embedding_detector (section 4.5.2)
```

> **No API key required**: all defense components validate structure and patterns; no LLM calls needed.

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
