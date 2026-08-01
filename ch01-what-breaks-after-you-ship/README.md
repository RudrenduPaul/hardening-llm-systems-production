# Chapter 1: What Breaks After You Ship

Build the **HardeningReport** self-diagnostic scorecard — a 25-question framework that surfaces hardening gaps before they become production incidents.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch01_notebook.ipynb`](ch01_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch01-what-breaks-after-you-ship/ch01_notebook.ipynb) | Interactive notebook: run the scorecard, visualize vector scores, export to JSON |
| [`ch01_scripts.py`](ch01_scripts.py) | Production-ready `HardeningReport` class — drop into any CI pipeline |

## What this chapter builds

- **HardeningReport** — 25-question scorecard across 5 hardening vectors (Hallucination, Prompt Injection, Output Safety, Observability, Governance)
- **Readiness bands** — HARDENED / REINFORCED / EXPOSED / VULNERABLE scoring with numeric cutoffs
- **Prioritized remediation plan** — sorted by risk, not by vector order
- **JSON export** — machine-readable output for governance dashboards and JIRA ticket generators
- **Score tracking over time** — simulate scorecard runs before launch, after model updates, after incidents
- **Threat model vocabulary (§1.9)** — system boundary (5 layers), asset inventory, actor taxonomy, and failure class table that every subsequent chapter assumes
- **scan_llm_stack** — scans requirements.txt against NIST NVD API v2 for known CVEs; returns a `StackScanReport` sorted by KEV status
- **tag_incident** — TF-IDF cosine-similarity tagger that maps a production incident description to one of four GenAI failure archetypes
- **classify_incident** — keyword-count classifier identifying which of four GenAI risk properties (non-determinism, open-ended output, emergent behavior, instruction-following) drove a given incident
- **generate_exposure_map** — generates the post-deployment exposure map for a team, detecting coverage gaps and single points of failure per hardening vector
- **HardeningReadinessReport** — versioned JSON readiness report built from scorecard answers; saved to dated filenames for CI comparison and Annex IV audit trails

## Prerequisites

- Python 3.10+
- No external packages — this chapter uses the standard library only

## Key concepts

| Concept | Description |
|---------|-------------|
| Hardening vectors | Five risk domains: Hallucination, Prompt Injection, Output Safety, Observability, Governance |
| Readiness band | Aggregate score bucketed into four operational tiers |
| Remediation priority | Re-order gaps by combined severity × coverage before assigning sprints |

## Running locally

```bash
cd ch01-what-breaks-after-you-ship
jupyter notebook ch01_notebook.ipynb
```
