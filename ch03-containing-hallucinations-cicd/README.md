# Chapter 3: Containing Hallucinations as a CI/CD-Blocking Metric

Wire the hallucination detection pipeline from Chapter 2 into a GitHub Actions CI gate — every pull request that touches a prompt, retrieval config, or model version is blocked when faithfulness drops below threshold.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch03_notebook.ipynb`](ch03_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch03-containing-hallucinations-cicd/ch03_notebook.ipynb) | Interactive notebook: run the CI gate locally, visualize pass/fail thresholds |
| [`ch03_scripts.py`](ch03_scripts.py) | `HallucinationCIGate`, `RegressionSuiteRunner`, `ThresholdPolicy` |

## What this chapter builds

- **HallucinationCIGate** — pass/fail gate that runs the combined scorer and exits with code 1 on threshold breach
- **RegressionSuiteRunner** — loads a JSON test suite, runs all test cases, reports aggregate and per-case results
- **ThresholdPolicy** — configurable strictness levels (strict / balanced / lenient) with justification
- **CI configuration templates** — ready-to-paste GitHub Actions YAML for hallucination gates
- **Baseline tracking** — saves golden baseline scores and alerts on regression, not just absolute failure

## Prerequisites

```bash
pip install deepeval matplotlib
```

> **No API key required** — the notebook uses mock scorers with pre-seeded distributions.

## Key concepts

| Concept | Description |
|---------|-------------|
| Regression gate | Blocks merges when faithfulness drops relative to baseline, not just below a fixed floor |
| Threshold policy | Separate policies for pre-prod, staging, and production environments |
| Golden baseline | Frozen score snapshot checked into the repo alongside the test suite |

## Running locally

```bash
cd ch03-containing-hallucinations-cicd
jupyter notebook ch03_notebook.ipynb
```
