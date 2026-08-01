# Chapter 6: Red-Teaming — Automated Validation Before Anyone Else Does

Build an automated red-team pipeline combining Garak (probe-based scanning), PyRIT (adversarial objective-driven attacks), and Promptfoo (YAML-driven regression evaluation) into a single CI-integrated workflow that runs before every production deployment.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch06_notebook.ipynb`](ch06_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/ch06-red-teaming/ch06_notebook.ipynb) | Interactive notebook: parse scan results, build scoring heatmap, generate CI test file |
| [`ch06_scripts.py`](ch06_scripts.py) | `run_garak_scan`/`parse_garak_report`, `run_pyrit_pair_attack`, `generate_promptfoo_config`/`run_promptfoo`, `RedTeamOrchestrator`, `LLMRedTeamScoringFramework`, `ci_red_team_gate` |

## What this chapter builds

- **run_garak_scan / parse_garak_report** — launches a Garak scan and parses the JSONL report into structured `GarakFinding` objects
- **run_pyrit_pair_attack** — runs PyRIT's PAIR (Prompt Automatic Iterative Refinement) adaptive jailbreak attack via `PAIROrchestrator`
- **generate_promptfoo_config / run_promptfoo** — writes a Promptfoo red-team YAML config and parses the JSON evaluation results
- **RedTeamOrchestrator** — normalizes Garak, PyRIT, and Promptfoo output into a shared `RedTeamFinding` schema and aggregates them into one `OrchestratorReport`
- **LLMRedTeamScoringFramework** — five-metric CVSS-equivalent severity scoring for each normalized finding
- **ci_red_team_gate** — evaluates an `OrchestratorReport` against CI policy thresholds and returns the CI exit code
- **AttackTag / TaggedTestCase / build_coverage_matrix** — three-dimensional attack tag schema and 36-cell coverage matrix generator ensuring systematic attack surface coverage across dimensions
- **RAGClient / test_retrieval_manipulation** — plants an adversarial document and verifies whether it surfaces in RAG retrieval
- **scan_cot_for_leakage / batch_scan_cot_traces** — chain-of-thought leakage scanner using sentence-transformers cosine similarity (falls back to Jaccard keyword similarity) to detect when reasoning traces surface system-prompt content
- **STEP_EXTRACTION_PROBES / run_step_extraction_probes** — probe set that forces a reasoning model to reveal its constraints in the CoT trace, then scans the trace for leakage

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
