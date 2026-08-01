# Hardening LLM Systems in Production: Companion Code

[![CI](https://github.com/RudrenduPaul/hardening-llm-systems-production/actions/workflows/test-notebooks.yml/badge.svg?branch=main)](https://github.com/RudrenduPaul/hardening-llm-systems-production/actions/workflows/test-notebooks.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Publisher](https://img.shields.io/badge/Publisher-Manning%20Publications-CC0000.svg)](https://www.manning.com)

| | |
|---|---|
| **Book** | *Hardening LLM Systems in Production: An Engineer's Playbook for Hallucinations, Prompt Injection, RAG Security, Agent Containment, and EU AI Act Compliance* |
| **Author** | Rudrendu Paul · [ORCID](https://orcid.org/0009-0008-0141-4690) · [LinkedIn](https://www.linkedin.com/in/rudrendupaul/) |
| **Co‑author** | Sourav Nandy · [LinkedIn](https://www.linkedin.com/in/souravnandy/) · [GitHub](https://github.com/Sourav-Nandy-ai) |
| **Publisher** | Manning Publications (forthcoming) |

Clone the companion code repository:

```bash
git clone https://github.com/RudrenduPaul/hardening-llm-systems-production.git
cd hardening-llm-systems-production/companion-code
```

---

## Prerequisites

- Python 3.10 or 3.11 recommended (`python --version  # requires 3.10+, tested on 3.11`)
- `pip install -r requirements.txt`
- `python -m spacy download en_core_web_lg` (needed for Presidio NER in Chapter 8)

All library versions in `requirements.txt` are pinned with `==` for full reproducibility. A weekly CI job flags version drift automatically; check the **Actions** tab for current freshness status.

---

## Table of contents

| Chapter | Notebook | Key scripts | Main topic |
|---------|----------|-------------|------------|
| [Ch 1: What Breaks After You Ship](ch01-what-breaks-after-you-ship/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch01-what-breaks-after-you-ship/ch01_notebook.ipynb) | [ch01_scripts.py](ch01-what-breaks-after-you-ship/ch01_scripts.py) | Self-diagnostic hardening scorecard + threat model vocabulary |
| [Ch 2: Detecting Hallucinations](ch02-detecting-hallucinations/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch02-detecting-hallucinations/ch02_notebook.ipynb) | [ch02_scripts.py](ch02-detecting-hallucinations/ch02_scripts.py) | deepeval + RAGAS hallucination detection pipeline |
| [Ch 3: Containing Hallucinations via CI/CD](ch03-containing-hallucinations-cicd/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch03-containing-hallucinations-cicd/ch03_notebook.ipynb) | [ch03_scripts.py](ch03-containing-hallucinations-cicd/ch03_scripts.py) | CI/CD hallucination gate with GitHub Actions |
| [Ch 4: Prompt Injection Defense](ch04-prompt-injection-defense/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch04-prompt-injection-defense/ch04_notebook.ipynb) | [ch04_scripts.py](ch04-prompt-injection-defense/ch04_scripts.py) | Defense-in-depth injection architecture |
| [Ch 5: RAG and Retrieval Security](ch05-rag-retrieval-security/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch05-rag-retrieval-security/ch05_notebook.ipynb) | [ch05_scripts.py](ch05-rag-retrieval-security/ch05_scripts.py) | Per-tenant RAG authorization + retrieval anomaly detection |
| [Ch 6: Red-Teaming: Automated Validation](ch06-red-teaming/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch06-red-teaming/ch06_notebook.ipynb) | [ch06_scripts.py](ch06-red-teaming/ch06_scripts.py) | Garak + PyRIT + Promptfoo CI scanning + reasoning-model trace testing |
| [Ch 7: Autonomous Agents: Scope, Containment, Monitoring](ch07-autonomous-agents-scope-containment-monitoring/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch07-autonomous-agents-scope-containment-monitoring/ch07_notebook.ipynb) | [ch07_scripts.py](ch07-autonomous-agents-scope-containment-monitoring/ch07_scripts.py) | MCP allowlists, sandboxed execution, tripwires, memory-poisoning defenses |
| [Ch 8: PII, Memorization, and Right-to-Erasure](ch08-pii-memorization-right-to-erasure/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch08-pii-memorization-right-to-erasure/ch08_notebook.ipynb) | [ch08_scripts.py](ch08-pii-memorization-right-to-erasure/ch08_scripts.py) | Presidio PII pipeline + memorization extraction defense + right-to-erasure |
| [Ch 9: Bias, Harmful Output, and Content Safety](ch09-bias-harmful-output-content-safety/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch09-bias-harmful-output-content-safety/ch09_notebook.ipynb) | [ch09_scripts.py](ch09-bias-harmful-output-content-safety/ch09_scripts.py) | Runtime content classification + counterfactual bias detection |
| [Ch 10: EU AI Act and NIST: Engineering Artifacts](ch10-eu-ai-act-nist-engineering-artifacts/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch10-eu-ai-act-nist-engineering-artifacts/ch10_notebook.ipynb) | [ch10_scripts.py](ch10-eu-ai-act-nist-engineering-artifacts/ch10_scripts.py) | Annex IV documentation generator + 15-day incident-response pipeline |
| [Ch 11: The Hardening Stack: PR-Gate](ch11-hardening-stack-pr-gate/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch11-hardening-stack-pr-gate/ch11_notebook.ipynb) | [ch11_scripts.py](ch11-hardening-stack-pr-gate/ch11_scripts.py) | Seven-layer hardening stack + Garak/PyRIT/Promptfoo CI scanning |

---

## Repository structure

Each chapter directory contains two files plus a `README.md`:

```
companion-code/
├── requirements.txt
├── .env.example
├── execute_notebooks.py
├── notebook_prep.py
├── ch01-what-breaks-after-you-ship/
│   ├── README.md
│   ├── ch01_notebook.ipynb
│   └── ch01_scripts.py
├── ch02-detecting-hallucinations/
├── ch03-containing-hallucinations-cicd/
├── ch04-prompt-injection-defense/
├── ch05-rag-retrieval-security/
├── ch06-red-teaming/
├── ch07-autonomous-agents-scope-containment-monitoring/
├── ch08-pii-memorization-right-to-erasure/
├── ch09-bias-harmful-output-content-safety/
├── ch10-eu-ai-act-nist-engineering-artifacts/
└── ch11-hardening-stack-pr-gate/
```

---

## Local setup

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy NER model (required for Presidio in Ch 8)
python -m spacy download en_core_web_lg

# Configure API keys
cp .env.example .env
# Edit .env with your credentials
```

---

## Running and re-running notebooks

All notebooks include demo data and mock modes. Most chapters run fully offline without API keys.

```bash
# Run all notebooks and save outputs
python3 execute_notebooks.py

# Run a specific chapter
python3 execute_notebooks.py ch01 ch04 ch10
```

Outputs are saved in-place to the `.ipynb` files. GitHub renders cell outputs directly in the browser. Colab users can re-run any cell to reproduce the output.

---

## Environment variables

Copy `.env.example` to `.env` and fill in your credentials. See the file for a complete key list by chapter. Never commit your `.env` file; it's already excluded via `.gitignore`.

The most commonly needed keys:

| Key | Chapters | Required? |
|-----|----------|-----------|
| `OPENAI_API_KEY` | 2, 3, 6, 9, 11 | Optional: mock mode used if absent |
| `ANTHROPIC_API_KEY` | 9 | Optional: mock mode used if absent |
| `PINECONE_API_KEY` | 5 | Optional: stub client used if absent |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | 7, 11 | Optional: in-memory mode if absent |

---

## CI freshness policy

A weekly GitHub Actions workflow executes all notebooks to verify they run without error. If a library update breaks a notebook, an issue is opened automatically. Check the **Actions** tab for current freshness status.

---

## Citation

Paul, R., & Nandy, S. (2026). *Hardening LLM Systems in Production*. Manning Publications.

---

## License

Code in this repository is licensed under the **MIT License**.
Book content (the manuscript) is copyright Rudrendu Paul, all rights reserved.
