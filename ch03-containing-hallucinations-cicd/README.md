# Chapter 3: Containing Hallucinations as a CI/CD-Blocking Metric

Wire the hallucination detection pipeline from Chapter 2 into a GitHub Actions CI gate: every pull request that touches a prompt, retrieval config, or model version is blocked when the hallucination rate crosses threshold. Also implements the output-level checks and the shadow-traffic harness that validate a guardrail change before it reaches every user.

## Main chapter code

| File | Description |
|------|-------------|
| [`ch03_notebook.ipynb`](ch03_notebook.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RudrenduPaul/hardening-llm-systems-production/blob/main/companion-code/ch03-containing-hallucinations-cicd/ch03_notebook.ipynb) | Interactive notebook: run the CI gate locally, sample self-consistency and claim decomposition, replay shadow traffic |
| [`ch03_scripts.py`](ch03_scripts.py) | `HallucinationCheckGate`, `hash_prompt` / `evaluate_prompt_version`, `PROMPT_BEFORE` / `PROMPT_AFTER` / `EXAMPLES`, `GateConfig` + `HallucinationCIGate`, `SelfConsistencyChecker`, `ClaimDecompositionPipeline`, `ShadowTrafficHarness`, `canary_window_sizing`, GitHub Actions YAML generator |

## What this chapter builds

- **HallucinationCheckGate** (Listing 3.2): the minimal CI gate: scores a test suite with a pluggable scorer function and calls `sys.exit(1)` when the mean score falls below `threshold`. Distinct from ch02's `HallucinationGate` wrapper (`run(pairs) -> dict`, `exit_code()`); this class doesn't subclass or extend it, it's a separate merge-blocking gate built on top of the same scorer contract.
- **hash_prompt / evaluate_prompt_version** (section 3.2.1): version-tracks a system prompt by content hash and evaluates it against the golden dataset, returning a pass/fail summary you can diff across prompt versions.
- **PROMPT_BEFORE / PROMPT_AFTER / EXAMPLES** (Listing 3.3, section 3.2.2): the before/after scope-limited prompt comparison with per-example annotated expected behavior.
- **GateConfig + HallucinationCIGate** (Listing 3.7): the complete CI/CD gate: configuration management, the re-run policy (re-scores a fresh sample slice when a run fails, requires `rerun_pass_count` passes out of `max_reruns + 1` attempts before blocking), baseline drift detection (flags a large rate shift as a model/provider-change warning instead of a silent pass or fail), and concurrency-bounded scoring (`_run_once` scores each sample slice on a `ThreadPoolExecutor` sized by `config.concurrency`).
- **SelfConsistencyChecker** (Listing 3.4): samples N completions for a question, normalizes each response, and takes a majority vote; flags the question when the agreement rate falls below `min_agreement`.
- **ClaimDecompositionPipeline** (Listing 3.5): decomposes a compound answer into atomic claims and scores each one against retrieved context with a keyword-overlap heuristic (a claim is supported when at least 40% of its tokens appear in the context).
- **ShadowTrafficHarness** (Listing 3.6): replays a batch of logged production traffic against a candidate configuration, scores both, and returns a `ShadowReport` with a promote/hold/block recommendation (both a human-readable string and a machine-readable `recommendation_code`).
- **canary_window_sizing** (section 3.6): two-proportion-z-test power analysis that converts a baseline rate, a detectable shift, and total traffic volume into the required sample size and canary window duration in hours.
- **GitHub Actions YAML generator**: `GITHUB_ACTIONS_YAML` / `print_github_actions_yaml()`, a ready-to-paste workflow matching manuscript Listing 3.1 (PR-to-main only, scoped to `src/`, `prompts/`, and `config/` changes).

## Prerequisites

```bash
pip install openai deepeval==0.21.71 ragas==0.2.15 scikit-learn scipy pydantic
```

> **No API key required for the demos**: `SelfConsistencyChecker`, `ClaimDecompositionPipeline`, and `evaluate_prompt_version` fall back to mock/rule-based generators, and `HallucinationCheckGate` / `HallucinationCIGate` ship with a deterministic mock scorer, when `OPENAI_API_KEY` isn't set. `canary_window_sizing` requires `scipy`.

## Key concepts

| Concept | Description |
|---------|-------------|
| Re-run policy | Absorbs LLM-as-judge flakiness: a single failing run doesn't block the merge on its own (section 3.7.2) |
| Baseline drift detection | Flags a model/provider version change as a recalibration prompt, not a PR rejection (section 3.7.3) |
| Shadow traffic | Compares a candidate guardrail configuration against production on real, already-logged traffic before any user sees its output (section 3.6) |

## Running locally

```bash
python ch03_scripts.py               # run every demo
python ch03_scripts.py --gate        # Listing 3.2 demo
python ch03_scripts.py --ci-gate     # Listing 3.7 demo: config, re-run policy, baseline drift
python ch03_scripts.py --consistency # SelfConsistencyChecker demo
python ch03_scripts.py --claims      # ClaimDecompositionPipeline demo
python ch03_scripts.py --shadow      # ShadowTrafficHarness demo
python ch03_scripts.py --prompt-version # prompt version tracking + before/after demo
python ch03_scripts.py --canary-sizing  # canary window power-analysis demo
python ch03_scripts.py --gen-yaml    # print the GitHub Actions YAML
```

```bash
cd ch03-containing-hallucinations-cicd
jupyter notebook ch03_notebook.ipynb
```
