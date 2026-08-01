# Chapter 9: Bias, Harmful Output, and Content Safety

Treat bias measurement and harmful output classification as two sides of the same engineering problem. Counterfactual probing and occupational association tests make demographic skew visible before it reaches users. Runtime classifiers, operational SLOs, and a CI/CD gate then block unsafe deploys with the same rigor.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch09_notebook.ipynb`](ch09_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch09-bias-harmful-output-content-safety/ch09_notebook.ipynb) | Interactive notebook: run counterfactual probes, measure bias gaps, run content classifiers, wire CI gate |
| [`ch09_scripts.py`](ch09_scripts.py) | `CounterfactualBiasProbe` (Listing 9.1), `OccupationalAssociationTest` (Listing 9.2), `LLMBiasJudge` (Listing 9.3), `HarmfulOutputTaxonomy` (Listing 9.4), NeMo Guardrails rails builder (Listing 9.5), Guardrails AI validator chain (Listing 9.6), `HarmfulContentSLO` (Listing 9.7), `HarmfulContentCIGate` (Listing 9.8) |

`ch09_notebook.ipynb` imports the classes above directly from `ch09_scripts.py` and exercises
each one against a small deterministic fake OpenAI client, so it runs end to end with no API
key or network access required.

## What this chapter builds

- **`CounterfactualBiasProbe`** (Listing 9.1) — counterfactual sentence-pair probing with a Welch t-test, sentiment gap, and word-count gap
- **`OccupationalAssociationTest`** (Listing 9.2) — pronoun-vs-BLS-baseline occupational association test with a per-occupation bias coefficient
- **`LLMBiasJudge`** (Listing 9.3) — three-run majority-vote LLM-as-judge with Cohen's kappa calibration against a human gold set
- **`HarmfulOutputTaxonomy`** (Listing 9.4) — severity-scored taxonomy covering all six Table 9.1 output types, mapped to OWASP LLM06/LLM09, GDPR Article 9/22, and EU AI Act Annex III/Article 52 liability
- **NeMo Guardrails rails builder** (Listing 9.5) — Colang conversation-flow policy + `check_hate_speech` action wired to the OpenAI moderation endpoint
- **Guardrails AI validator chain** (Listing 9.6) — `ToxicLanguage` + `DetectPII` validators with reask logic
- **`HarmfulContentSLO`** (Listing 9.7) — three-tier degradation ladder (block list, ML classifier, LLM judge) with P99 latency budgets and per-severity fail-safe/fail-open dispatch
- **`HarmfulContentCIGate`** (Listing 9.8) — unified release gate blocking on harmful fraction, counterfactual bias gap, occupational bias coefficient, and judge calibration kappa

## Prerequisites

```bash
# Required for Listings 9.1-9.4, 9.7, 9.8
pip install openai>=1.30.0 textblob>=0.17.0 scipy>=1.11.0 numpy>=1.26.0 scikit-learn>=1.3.0

# Optional, only needed to run Listing 9.5 / 9.6 directly
pip install nemoguardrails>=0.9.0 guardrails-ai>=0.4.0
```

## Running locally

```bash
# Run the full CI/CD gate against synthetic demo data (no API key needed)
python ch09_scripts.py --mode gate
# exits 1 if harmful_fraction > --harmful-threshold OR any bias gap/coefficient
# exceeds --bias-gap / the taxonomy's configured thresholds

# Run the interactive notebook (no API key needed -- see the fake-client cell)
jupyter notebook ch09_notebook.ipynb
```
