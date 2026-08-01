# Chapter 10: EU AI Act and NIST — Engineering Artifacts, Incident Response, and Regulatory Notification

Generate the Annex IV technical documentation required by the EU AI Act for high-risk AI systems, map controls to the NIST AI RMF, and produce a machine-readable compliance artifact that feeds into audit workflows and model registry entries.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch10_notebook.ipynb`](ch10_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch10-eu-ai-act-nist-engineering-artifacts/ch10_notebook.ipynb) | Interactive notebook: generate Annex IV document, map NIST controls, export compliance package |
| [`ch10_scripts.py`](ch10_scripts.py) | `AnnexIVPackage`, `NISTAI6001Tracker`, `ProvenanceRecord`/`create_provenance_record`, `TamperEvidentAuditLog`, `generate_dual_framework_report` |

## What this chapter builds

- **AnnexIVPackage** (Listing 10.1) — assembles the six engineering-facing Annex IV sections into a structured package; `completeness_score()` checks both field population and that referenced artifact files exist on disk
- **check_annex_iv_completeness()** (Listing 10.2) — CI/CD gate that `sys.exit(1)`s a merge or deploy when required fields are missing or referenced artifacts aren't on disk
- **create_provenance_record()** / **ProvenanceRecord** (Listing 10.3) — HMAC-SHA256-signed record linking a model output to its model version, prompt template version, and input/output hashes; `verify_provenance_record()` re-derives the signature to detect tampering
- **TamperEvidentAuditLog** (Listing 10.4) — append-only, chained-hash audit log; `verify_integrity()` returns `{"status": "broken", "broken_at_entry": N}` pointing at the first tampered entry
- **NistTriageReport** (Listing 10.4b) — prioritizes the 20 NIST AI 600-1 actions from section 10.5.1 by cluster; `gap_summary()` is the sprint backlog
- **NISTAI6001Tracker** (Listing 10.5) — tracks per-control implementation status and evidence paths; `gap_report()` returns a coverage percentage plus the list of open gaps
- **generate_dual_framework_report()** (Listing 10.6) — merges Annex IV completeness and NIST coverage into one report with an `overall_status` field (80% NIST coverage threshold)
- **PostMarketMonitoringReport** (Listing 10.8c) — monthly EU AI Act post-market monitoring report; `requires_regulatory_notification()` flags any P0 incident for legal review
- **IncidentEscalation** — tracks EU AI Act Article 73 notification timelines (standard 15-day, 2-day, 10-day windows); emits `sla_status()` for monitoring dashboards showing breached, warning, and complete phases
- **Article73NotificationPackage** — serializes the 72h and 15-day documentation packages with `is_72h_complete()` and `is_15d_complete()` completeness flags; `days_until_deadline()` drives daily alerting
- **IncidentClassifier** — classifies incidents as P0/P1/P2 using chapter 7 telemetry fields (`tripwire_fired`, `cusum_alert`, `cognitive_degradation_level`); starts the Article 73 clock on P0; returns `ClassificationResult` with rationale and SLA target
- **ContainmentRunbook** — executes Steps 1–4 of the containment playbook (fallback routing, memory freeze, credential revocation, state snapshot) with independent `try/except` per step; produces a `fully_contained` flag without aborting on partial failure

## Prerequisites

```bash
pip install pyyaml
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
