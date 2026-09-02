# Chapter 5: RAG and Retrieval Security (The Largest New Attack Surface)

Secure a Retrieval-Augmented Generation pipeline against namespace collision, cross-tenant data leakage, and adversarial poisoning: using per-tenant namespacing, embedding anomaly detection, and CUSUM statistical monitoring of retrieval patterns.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch05_notebook.ipynb`](ch05_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch05-rag-retrieval-security/ch05_notebook.ipynb) | Interactive notebook: namespace enforcement, embedding anomaly detection, CUSUM monitoring |
| [`ch05_scripts.py`](ch05_scripts.py) | `TenantScopedPineconeClient`, `EmbeddingAnomalyDetector`, `DefenseInDepthRetrievalPipeline`, `CUSUMRetrievalAnomalyDetector`, `CUSUMRetrievalMonitor`, `SanitizingDocumentLoader`, `InjectionPatternDetector` (Haystack) |

## What this chapter builds

- **TenantScopedPineconeClient**: SHA-256 deterministic namespace derivation; prevents tenant A from querying tenant B's vectors, with a post-filter safety net
- **EmbeddingAnomalyDetector**: Tukey-fence anomaly detection on query embedding L2 norm; flags adversarially crafted vectors
- **DefenseInDepthRetrievalPipeline** / **secure_retrieve**: layers embedding-anomaly detection, injection scanning, tenant-scoped retrieval, sanitization, and intent reranking
- **CUSUMRetrievalAnomalyDetector** / **CUSUMRetrievalMonitor**: CUSUM change detection on retrieval similarity scores and per-session cross-category rate; alerts on poisoning attempts
- **SanitizingDocumentLoader**: strips HTML, detects injection patterns, and redacts flagged content before ingestion (LangChain)
- **InjectionPatternDetector**: Haystack component wrapping the same detection/redaction logic as `SanitizingDocumentLoader`; `haystack-ai` is an optional import, so the module loads without it installed
- **agentic_retrieve_with_loop_defense**: depth-limited, provenance-logged retrieval loop for agentic RAG
- **StubPineconeClient**: in-memory Pinecone stub for offline testing and CI execution
- **Retrieval security test suite** (`ch05_retrieval_security_tests.py`): pytest file covering embedding anomaly detection, injection blocking, CUSUM alerting, and namespace isolation

## Prerequisites

```bash
pip install "numpy>=1.26.0,<2.0" matplotlib pydantic>=2.0
```

> **No live Pinecone required**: all notebook demos use the in-memory `StubPineconeClient`. Set `PINECONE_API_KEY` in `.env` for live execution.

## Key concepts

| Concept | Description |
|---------|-------------|
| Tenant namespace | Cryptographically derived index partition; namespace leakage is detectable, not just policy-blocked |
| CUSUM | Cumulative sum control chart for sequential anomaly detection on score time series |

## Running locally

```bash
cd ch05-rag-retrieval-security
jupyter notebook ch05_notebook.ipynb
```
