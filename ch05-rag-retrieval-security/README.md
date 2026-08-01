# Chapter 5: RAG and Retrieval Security — The Largest New Attack Surface

Secure a Retrieval-Augmented Generation pipeline against namespace collision, cross-tenant data leakage, and adversarial poisoning — using per-tenant namespacing, cryptographic provenance tags, and statistical anomaly detection on retrieval patterns.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch05_notebook.ipynb`](ch05_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch05-rag-retrieval-security/ch05_notebook.ipynb) | Interactive notebook: namespace enforcement, provenance tagging, CUSUM anomaly detection |
| [`ch05_scripts.py`](ch05_scripts.py) | `TenantNamespaceEnforcer`, `ProvenanceTaggedRetriever`, `CUSUMRetrievalMonitor` |

## What this chapter builds

- **TenantNamespaceEnforcer** — SHA-256 deterministic namespace derivation; prevents tenant A from querying tenant B's vectors
- **ProvenanceTaggedRetriever** — signs every document chunk at ingest time; verifies signature before context injection
- **CUSUMRetrievalMonitor** — CUSUM change detection on per-tenant retrieval score distributions; alerts on poisoning attempts
- **StubPineconeClient** — in-memory Pinecone stub for offline testing and CI execution
- **Retrieval security test suite** — auto-generated pytest file covering namespace bypass, signature tampering, score anomalies

## Prerequisites

```bash
pip install numpy matplotlib pydantic>=2.0
```

> **No live Pinecone required** — all notebook demos use the in-memory `StubPineconeClient`. Set `PINECONE_API_KEY` in `.env` for live execution.

## Key concepts

| Concept | Description |
|---------|-------------|
| Tenant namespace | Cryptographically derived index partition; namespace leakage is detectable, not just policy-blocked |
| Provenance tag | HMAC signature on chunk metadata; detects poisoned documents inserted after ingest |
| CUSUM | Cumulative sum control chart for sequential anomaly detection on score time series |

## Running locally

```bash
cd ch05-rag-retrieval-security
jupyter notebook ch05_notebook.ipynb
```
