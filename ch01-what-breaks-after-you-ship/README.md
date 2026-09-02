# Chapter 1: What Breaks After You Ship

Build the **HardeningReport** self-diagnostic scorecard: a 25-question framework that surfaces hardening gaps before they become production incidents.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch01_notebook.ipynb`](ch01_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch01-what-breaks-after-you-ship/ch01_notebook.ipynb) | Interactive notebook: run the scorecard, visualize vector scores, export to JSON |
| [`ch01_scripts.py`](ch01_scripts.py) | Production-ready `HardeningReport` class: drop into any CI pipeline |

## What this chapter builds

- **HardeningReport**: 25-question scorecard across 5 hardening vectors (Hallucination containment, Adversarial hardening, Agentic safety, Data leakage prevention, Compliance readiness: section 1.7)
- **Exposure tiers**: Critical exposure (0-10) / Partial hardening (11-18) / Defensible posture (19-25), the 3-tier scoring from section 1.7.3
- **Prioritized remediation plan**: sorted by vector score, not by vector order
- **JSON export**: machine-readable output for governance dashboards and JIRA ticket generators
- **Score tracking over time**: simulate scorecard runs before launch, after model updates, after incidents
- **Threat model vocabulary (§1.8)**: system boundary (5 layers), asset inventory, actor taxonomy, and failure class table that every subsequent chapter assumes
- **scan_llm_stack**: scans requirements.txt against NIST NVD API v2 for known CVEs; returns a `StackScanReport` sorted by KEV status
- **tag_incident**: TF-IDF cosine-similarity tagger that maps a production incident description to one of four GenAI failure archetypes
- **classify_incident**: keyword-count classifier identifying which of four GenAI risk properties (non-determinism, open-ended output, emergent behavior, instruction-following) drove a given incident
- **generate_exposure_map**: generates the post-deployment exposure map for a team, detecting coverage gaps and single points of failure per hardening vector
- **HardeningReadinessReport**: versioned JSON readiness report built directly from a completed `HardeningReport`'s own scores and exposure tier (Listing 1.6, section 1.7.5); saved to dated filenames for CI comparison and Annex IV audit trails

## Prerequisites

- Python 3.9+
- No external packages: this chapter uses the standard library only

## Key concepts

| Concept | Description |
|---------|-------------|
| Hardening vectors | Five risk domains: Hallucination containment, Adversarial hardening, Agentic safety, Data leakage prevention, Compliance readiness |
| Exposure tier | Aggregate 0-25 score bucketed into three tiers: Critical exposure, Partial hardening, Defensible posture |
| Remediation priority | Re-order gaps by vector score (lowest first) before assigning sprints |

## Running locally

```bash
cd ch01-what-breaks-after-you-ship
jupyter notebook ch01_notebook.ipynb
```
