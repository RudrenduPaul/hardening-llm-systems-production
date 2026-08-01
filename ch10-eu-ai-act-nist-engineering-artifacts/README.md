# Chapter 10: EU AI Act and NIST — Engineering Artifacts, Incident Response, and Regulatory Notification

Generate the Annex IV technical documentation required by the EU AI Act for high-risk AI systems, map controls to the NIST AI RMF, and produce a machine-readable compliance artifact that feeds into audit workflows and model registry entries.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch10_notebook.ipynb`](ch10_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch10-eu-ai-act-nist-engineering-artifacts/ch10_notebook.ipynb) | Interactive notebook: generate Annex IV document, map NIST controls, export compliance package |
| [`ch10_scripts.py`](ch10_scripts.py) | `AnnexIVGenerator`, `NISTControlMapper`, `ProvenanceRecorder`, `CompliancePackageExporter` |

## What this chapter builds

- **AnnexIVGenerator** — fills all 9 mandatory sections of EU AI Act Annex IV technical documentation from a model card YAML
- **NISTControlMapper** — maps hardening controls implemented in Chapters 1–9 to NIST AI RMF functions (GOVERN / MAP / MEASURE / MANAGE)
- **ProvenanceRecorder** — HMAC-signed immutable log of all system changes; satisfies Article 12 (record-keeping)
- **CompliancePackageExporter** — bundles Annex IV document, NIST mapping, provenance log, and model card into a dated ZIP archive
- **Compliance dashboard** — summary table of implemented vs. required controls by regulation
- **IncidentEscalation** — tracks EU AI Act Article 73 notification timelines (standard 15-day, 2-day, 10-day windows); emits `sla_status()` for monitoring dashboards showing breached, warning, and complete phases
- **Article73NotificationPackage** — serializes the 72h and 15-day documentation packages with `is_72h_complete()` and `is_15d_complete()` completeness flags; `days_until_deadline()` drives daily alerting
- **IncidentClassifier** — classifies incidents as P0/P1/P2 using chapter 7 telemetry fields (`tripwire_fired`, `cusum_alert`, `cognitive_degradation_level`); starts the Article 73 clock on P0; returns `ClassificationResult` with rationale and SLA target
- **ContainmentRunbook** — executes Steps 1–4 of the containment playbook (fallback routing, memory freeze, credential revocation, state snapshot) with independent `try/except` per step; produces a `fully_contained` flag without aborting on partial failure

## Prerequisites

```bash
pip install pyyaml pydantic>=2.0
```

> **No API key required** — this chapter uses only dataclasses, YAML, and cryptographic stdlib modules.

## Key concepts

| Concept | Description |
|---------|-------------|
| Annex IV | EU AI Act technical documentation requirement for high-risk AI systems (Article 11) |
| NIST AI RMF | Four-function framework: Govern, Map, Measure, Manage — maps to concrete engineering controls |
| Provenance log | Immutable, signed record of model versions, prompt changes, and configuration updates |

## Running locally

```bash
cd ch10-eu-ai-act-nist-engineering-artifacts
jupyter notebook ch10_notebook.ipynb
```
