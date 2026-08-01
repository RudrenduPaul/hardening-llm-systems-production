# Chapter 6: Red-Teaming — Automated Validation Before Anyone Else Does

Build an automated red-team pipeline combining Garak (probe-based scanning), PyRIT (adversarial objective-driven attacks), and Promptfoo (YAML-driven regression evaluation) into a single CI-integrated workflow that runs before every production deployment.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch06_notebook.ipynb`](ch06_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch06-red-teaming/ch06_notebook.ipynb) | Interactive notebook: parse scan results, build scoring heatmap, generate CI test file |
| [`ch06_scripts.py`](ch06_scripts.py) | `GarakScanParser`, `PyRITResultAnalyzer`, `PromptfooEvaluator`, `RedTeamCIGate` |

## What this chapter builds

- **GarakScanParser** — parses Garak JSON report; classifies probes by risk level; generates per-category summary
- **PyRITResultAnalyzer** — extracts jailbreak attempt patterns; computes attack success rates by strategy
- **PromptfooEvaluator** — structured output verifier for YAML-defined evaluation suites
- **RedTeamCIGate** — aggregates all three scanners; fails CI when any risk category exceeds threshold
- **Scoring heatmap** — risk-by-category matrix visualization for security review meetings
- **CI test file generator** — auto-generates pytest test stubs from Garak probe categories
- **AttackTag / TaggedTestCase / build_coverage_matrix** — three-dimensional attack tag schema and 36-cell coverage matrix generator ensuring systematic attack surface coverage across dimensions
- **scan_cot_for_leakage / batch_scan_cot_traces** — chain-of-thought leakage scanner using sentence-transformers cosine similarity to detect when reasoning traces surface system-prompt content

## Prerequisites

```bash
pip install matplotlib
```

> **No API key required** — all notebook demos use pre-built mock scan reports (`MOCK_GARAK_REPORT`, `MOCK_PYRIT_RESULT`). Set `OPENAI_API_KEY` and install `garak`, `pyrit`, `promptfoo` for live red-teaming.

## Key concepts

| Concept | Description |
|---------|-------------|
| Probe-based scanning (Garak) | Systematic coverage of known vulnerability classes across model behavior |
| Objective-driven attack (PyRIT) | Attacker agent iterates prompts toward a harmful objective; measures steps to success |
| Regression evaluation (Promptfoo) | YAML test suites that verify model still refuses harmful requests after model update |

## Running locally

```bash
cd ch06-red-teaming
jupyter notebook ch06_notebook.ipynb
```
