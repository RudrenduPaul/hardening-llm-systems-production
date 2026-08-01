# Chapter 8: PII, Memorization, and Right-to-Erasure

Build a PII-guarded LLM pipeline using Microsoft Presidio, a memorization probe that detects training data regurgitation, a dual-output scanner for reasoning-model chain-of-thought leakage, and a GDPR/CCPA right-to-erasure pipeline for vector stores — then wire PII and memorization thresholds into a release-blocking CI gate.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch08_notebook.ipynb`](ch08_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch08-pii-memorization-right-to-erasure/ch08_notebook.ipynb) | Interactive notebook: scan and redact PII, run the memorization probe, run the dual-output reasoning-trace scanners, exercise the erasure ledger, run the CI gate |
| [`ch08_scripts.py`](ch08_scripts.py) | `build_analyzer` / `analyze_text` (Listing 8.1), `build_anonymizer` / `anonymize_text` (Listing 8.2), `QuasiIdentifierTracker` (Listing 8.2a), `PIIGuardedLLMPipeline` (Listing 8.3), `memorization_probe` (Listing 8.4), `scan_reasoning_model_output` (Listing 8.5), `CoTFilter` (Listing 8.5a), `scan_o3_response` (Listing 8.5b), `execute_right_to_erasure` (Listing 8.6), `ErasureLedger` (Listing 8.6a), `pgvector_erasure` / `weaviate_erasure` (Listing 8.6b), `PIIGateConfig` / `run_pii_gate` (Listing 8.7) |

## What this chapter builds

- **`build_analyzer` / `analyze_text`** (Listing 8.1) — Presidio `AnalyzerEngine` with a custom `PATIENT_ID` pattern recognizer (`MRN-\d{7}`) alongside the default entity types
- **`build_anonymizer` / `anonymize_text`** (Listing 8.2) — per-entity-type `OperatorConfig` mapping: replace, mask, or SHA-256 hash depending on entity
- **`QuasiIdentifierTracker`** (Listing 8.2a) — session-level combination scoring for quasi-identifiers; fires `HIGH` risk when 3+ distinct quasi-id types reference the same named entity
- **`PIIGuardedLLMPipeline`** (Listing 8.3) — wraps any LLM call with Presidio input redaction and output scanning, returning a `PipelineResult` audit trail
- **`memorization_probe`** (Listing 8.4) — Carlini-style extraction probe: greedy decoding (temperature=0), `difflib.SequenceMatcher` similarity, best-of-`n_samples` scoring
- **`scan_reasoning_model_output`** (Listing 8.5) — dual-output PII scanner for Claude extended-thinking responses; scans both the `thinking` block and the final answer independently
- **`CoTFilter`** (Listing 8.5a) — scans chain-of-thought traces for system-prompt-leak signals and Presidio PII before the trace reaches the user; returns a sanitized trace or full suppression
- **`scan_o3_response`** (Listing 8.5b) — dual-output PII scanner for OpenAI o3/o4-mini via the responses API; flags PII present in the reasoning summary but not the final output
- **`execute_right_to_erasure`** (Listing 8.6) — user-scoped Pinecone vector deletion against a `document_registry` mapping, returning an `ErasureResult` audit record
- **`ErasureLedger`** (Listing 8.6a) — JSON-persisted `doc_id -> embedding_ids` ledger implementing the two-phase erasure protocol; `execute_erasure()` returns a GDPR Article 5(2) `ErasureReport`
- **`pgvector_erasure` / `weaviate_erasure`** (Listing 8.6b) — right-to-erasure implementations for pgvector (`DELETE ... WHERE user_id`) and Weaviate (`delete_many` with a property filter)
- **`PIIGateConfig` / `run_pii_gate`** (Listing 8.7) — CI/CD gate combining PII rate and memorization rate thresholds; exits with code 1 on failure

## Prerequisites

```bash
pip install presidio-analyzer==2.2.354 presidio-anonymizer==2.2.354 \
            openai>=1.30.0,<2.0 anthropic>=0.25.0,<1.0 pinecone-client==4.1.0 \
            psycopg2-binary==2.9.9 weaviate-client==4.5.4
python -m spacy download en_core_web_lg    # required for Presidio NER
```

> **No API keys required for most demos.** `build_analyzer()` / `build_anonymizer()` degrade to `None` and log a warning if `presidio-analyzer` / `presidio-anonymizer` aren't installed, and `PIIGuardedLLMPipeline` / `memorization_probe` accept any injected client, so the notebook exercises them with a stub client. `scan_reasoning_model_output()` and `scan_o3_response()` instantiate a live `anthropic.Anthropic()` / `openai.OpenAI()` client internally and require `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` for live execution — those notebook cells are commented out by default.

## Key concepts

| Concept | Description |
|---------|-------------|
| Memorization rate | Fraction of probes where the best-of-`n_samples` completion exceeds `similarity_threshold` (default 0.85) similarity to the reference text |
| Quasi-identifier combination | 3+ distinct quasi-identifier types (location, date, employer, occupation, age bracket, etc.) observed about the same named entity within a session |
| PII rate | Fraction of test outputs where `analyze_text()` returns at least one entity above the 0.7 confidence threshold — production SLO is typically <2% (`PIIGateConfig.max_pii_rate`) |

## Running locally

```bash
cd ch08-pii-memorization-right-to-erasure
python ch08_scripts.py          # runs the PII gate CLI entry point (exits 1 on failure)
python -m pytest ch08_scripts.py -v   # runs the pytest test stubs
jupyter notebook ch08_notebook.ipynb
```
