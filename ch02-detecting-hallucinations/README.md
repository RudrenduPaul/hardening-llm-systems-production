# Chapter 2: Detecting Hallucinations Before Your Users Do

Build a production hallucination detection pipeline that combines deepeval's `HallucinationMetric` and RAGAS faithfulness scoring into a unified ensemble scorer with statistical reliability guarantees.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch02_notebook.ipynb`](ch02_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch02-detecting-hallucinations/ch02_notebook.ipynb) | Interactive notebook: run hallucination metrics, visualize score distributions, compute kappa |
| [`ch02_scripts.py`](ch02_scripts.py) | `HallucinationMetric`, `RAGASFaithfulnessScorer`, `CombinedHallucinationScorer` |

## What this chapter builds

- **HallucinationMetric** — thin wrapper around deepeval with normalized interface
- **RAGASFaithfulnessScorer** — sentence-level claim decomposition and entailment checking
- **CombinedHallucinationScorer** — ensemble averaging with configurable weights
- **Inter-rater reliability** — Cohen's kappa validates metric agreement before deploying
- **Statistical power analysis** — minimum sample size calculator for hallucination regression suites
- **Score distribution visualization** — histogram + violin plots for production dashboards

## Prerequisites

```bash
pip install deepeval ragas scikit-learn scipy
```

> **No API key required for the notebook demo** — all examples use mock scorers that simulate realistic score distributions.

## Key concepts

| Concept | Description |
|---------|-------------|
| Faithfulness | Does the output contain only claims supported by the retrieved context? |
| Answer relevancy | Does the output address the user's actual question? |
| Cohen's kappa | Agreement statistic for validating that two metrics agree beyond chance |
| Power analysis | How many test cases you need to detect a 5-point drop in faithfulness |

## Running locally

```bash
cd ch02-detecting-hallucinations
jupyter notebook ch02_notebook.ipynb
```
