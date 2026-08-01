# Chapter 8: PII, Memorization, and Right-to-Erasure

Build a PII-guarded LLM pipeline using Microsoft Presidio, a memorization probe that detects training data regurgitation, and a CI gate that fails on unacceptable PII leakage — then wire a GDPR-compliant right-to-erasure pipeline for on-demand vector store cleanup.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch08_notebook.ipynb`](ch08_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch08-pii-memorization-right-to-erasure/ch08_notebook.ipynb) | Interactive notebook: scan for PII, measure memorization, run bias probes, export CI report |
| [`ch08_scripts.py`](ch08_scripts.py) | `PIIPipeline`, `MemorizationProbe`, `RightToErasurePipeline`, `PIICIGate` |

## What this chapter builds

- **Custom PATIENT_ID recognizer** — extends Presidio with domain-specific entity types
- **PIIPipeline** — analyze + anonymize with per-entity operator mapping (replace / redact / hash)
- **MemorizationProbe** — difflib similarity scan on model completions; flags training data regurgitation above threshold
- **Dual-output PII scanner** — handles Claude extended-thinking models where reasoning and answer differ
- **RightToErasurePipeline** — deletes PII embeddings from vector stores on GDPR Article 17 requests with audit log
- **PIICIGate** — gate that exits with code 1 if PII recall rate exceeds the configured SLO threshold
- **QuasiIdentifierTracker** — session-level combination scoring for quasi-identifiers; fires when 3+ distinct quasi-id types reference the same named entity
- **CoTFilter** — scans reasoning traces for system-prompt signal leakage and Presidio PII before the trace reaches the user; returns sanitized trace or full suppression
- **ErasureLedger** — JSON-persisted document-to-embedding-IDs ledger; `execute_erasure()` handles both deletion phases atomically and returns a GDPR Article 5(2) audit record

## Prerequisites

```bash
pip install scipy matplotlib textblob
pip install presidio-analyzer presidio-anonymizer  # optional — mocked if missing
python -m spacy download en_core_web_lg            # required for Presidio NER
```

> **No Anthropic API key required for most demos** — extended-thinking demo uses mock output. Set `ANTHROPIC_API_KEY` for live execution.

## Key concepts

| Concept | Description |
|---------|-------------|
| Memorization rate | Fraction of completions that reproduce >60% of a training sequence verbatim |
| Counterfactual test | Compare outputs when only demographic attribute changes; statistically significant differences indicate bias |
| PII recall rate | Fraction of requests where PII appears in model output after scanning — production SLO is typically <1% |

## Running locally

```bash
cd ch08-pii-memorization-right-to-erasure
jupyter notebook ch08_notebook.ipynb
```
