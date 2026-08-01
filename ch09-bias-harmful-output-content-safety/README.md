# Chapter 9: Bias, Harmful Output, and Content Safety

Treat bias measurement and harmful output classification as two sides of the same engineering problem. Counterfactual probing and occupational association tests make demographic skew visible before it reaches users. Runtime classifiers, operational SLOs, and a CI/CD gate then block unsafe deploys with the same rigor.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch09_notebook.ipynb`](ch09_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch09-bias-harmful-output-content-safety/ch09_notebook.ipynb) | Interactive notebook: run counterfactual probes, measure bias gaps, run content classifiers, wire CI gate |
| [`ch09_scripts.py`](ch09_scripts.py) | `HarmfulOutputTaxonomy`, `NeMoContentClassifier`, `GuardrailsAIValidator`, `HarmfulContentSLO`, `CounterfactualBiasProbe`, `BiasJudgeCalibrator`, `HarmfulContentCIGate` |

## What this chapter builds

- **HarmfulOutputTaxonomy** — severity-scored dataclass mapping output types to OWASP LLM09/LLM06, GDPR Article 22, and EU AI Act Annex III liability
- **NeMoContentClassifier** — Colang policy + async Python launcher for hate-speech and harmful-content detection
- **GuardrailsAIValidator** — ToxicLanguage + DetectPII validator chain with reask logic
- **HarmfulContentSLO** — P99 latency budget enforcement with fail-safe/fail-open policy dispatch
- **CounterfactualBiasProbe** — counterfactual sentence pairs + occupational association tests with statistical gap scoring
- **BiasJudgeCalibrator** — Cohen's kappa + classification report for LLM-as-judge recalibration
- **HarmfulContentCIGate** — unified gate blocking harmful fraction + calibration drift + bias gap

## Prerequisites

```bash
pip install nemoguardrails==0.9.1 guardrails-ai==0.4.5 openai==1.30.0 scikit-learn==1.5.0
```

## Running locally

```bash
# Run the full CI/CD gate
python ch09_scripts.py --mode gate
# exits 1 if harmful_fraction > threshold OR bias_gap > 0.15

cd ch09-bias-harmful-output-content-safety
jupyter notebook ch09_notebook.ipynb
```
