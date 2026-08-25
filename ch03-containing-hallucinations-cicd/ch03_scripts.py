"""
Chapter 3: Containing Hallucinations as a CI/CD-Blocking Metric
Hardening LLM Systems in Production — Companion Code
Author: Rudrendu Paul | https://orcid.org/0009-0008-0141-4690

Implements:
  - HallucinationCheckGate: CI gate class with sys.exit(1) on failure (Listing 3.2)
  - hash_prompt / evaluate_prompt_version: prompt version tracking against
    the golden dataset (section 3.2.1)
  - PROMPT_BEFORE / PROMPT_AFTER / EXAMPLES: before/after prompt comparison
    with annotated expected behavior (Listing 3.3, section 3.2.2)
  - GateConfig + HallucinationCIGate: the complete CI/CD gate — config
    management, the re-run policy, baseline drift detection, and
    concurrency-bounded scoring (Listing 3.7, sections 3.7.1-3.7.3)
  - SelfConsistencyChecker: sample N completions, measure agreement
  - ClaimDecompositionPipeline: decompose compound answers into atomic claims
  - ShadowTrafficHarness: replay production traffic against a candidate model
  - canary_window_sizing: power-analysis-backed canary window duration
    (section 3.6)
  - GitHub Actions YAML generation helper

Requirements:
    openai>=1.0.0,<2.0
    deepeval==0.21.71
    ragas==0.2.15
    scikit-learn>=1.3.0,<2.0
    scipy>=1.11.0,<2.0
    pydantic==2.7.1        # optional — GateConfig falls back to a dataclass
                            # if pydantic isn't installed, so the script
                            # stays importable without the full ML stack

Usage:
    python ch03_scripts.py
    python ch03_scripts.py --gate          # Run CI gate demo (Listing 3.2)
    python ch03_scripts.py --ci-gate       # Run full CI/CD gate demo (Listing 3.7):
                                            # config, re-run policy, baseline drift
    python ch03_scripts.py --consistency   # Run self-consistency demo
    python ch03_scripts.py --claims        # Run claim decomposition demo
    python ch03_scripts.py --shadow        # Run shadow traffic demo
    python ch03_scripts.py --prompt-version # Run prompt version tracking demo
    python ch03_scripts.py --canary-sizing # Run canary window sizing demo
    python ch03_scripts.py --gen-yaml      # Print the GitHub Actions YAML
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Optional imports — system stays importable without the ML stack
# ---------------------------------------------------------------------------

try:
    import openai as _openai_module
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from pydantic import BaseModel as _PydanticBaseModel
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False

try:
    from scipy.stats import norm as _scipy_norm
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. HallucinationCheckGate — the CI gate class
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Outcome of a single CI gate run."""
    passed: bool
    score: float
    threshold: float
    n_samples: int
    fail_count: int
    details: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] score={self.score:.4f} threshold={self.threshold} "
            f"n={self.n_samples} failures={self.fail_count}"
        )


class HallucinationCheckGate:
    """
    CI gate that enforces a hallucination score threshold on a test suite.

    Designed to run inside GitHub Actions or any CI system. If the mean
    faithfulness score across the test suite falls below `threshold`, the
    gate calls sys.exit(1), failing the build.

    Parameters
    ----------
    threshold : float
        Minimum acceptable mean faithfulness score (0–1). Default 0.80.
    scorer : callable, optional
        A function (question, answer, context) -> float. If None, uses
        the mock scorer suitable for dry runs and unit tests.
    exit_on_fail : bool
        If True (default), calls sys.exit(1) on gate failure. Set to False
        in tests to avoid exiting the process.

    Usage in CI script
    ------------------
        gate = HallucinationCheckGate(threshold=0.80)
        results = gate.run(test_suite)
        gate.report(results)
        gate.enforce(results)  # sys.exit(1) if score < threshold
    """

    def __init__(
        self,
        threshold: float = 0.80,
        scorer: Optional[Callable] = None,
        exit_on_fail: bool = True,
    ) -> None:
        self.threshold = threshold
        self.scorer = scorer or self._mock_scorer
        self.exit_on_fail = exit_on_fail

    # ------------------------------------------------------------------
    # Core gate logic
    # ------------------------------------------------------------------

    def run(self, test_suite: list[dict]) -> GateResult:
        """
        Score each sample in the test suite and aggregate.

        Parameters
        ----------
        test_suite : list[dict]
            Each dict must contain: 'question', 'answer', 'context' (list[str]).
            Optional key: 'ground_truth' (str).

        Returns
        -------
        GateResult with per-sample details and aggregate score.
        """
        details = []
        for sample in test_suite:
            try:
                score = self.scorer(
                    sample["question"],
                    sample["answer"],
                    sample["context"],
                )
                details.append({
                    "question": sample["question"][:80],
                    "score": round(float(score), 4),
                    "passed": score >= self.threshold,
                })
            except Exception as exc:
                details.append({
                    "question": sample.get("question", "")[:80],
                    "score": 0.0,
                    "passed": False,
                    "error": str(exc),
                })

        scores = [d["score"] for d in details]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        fail_count = sum(1 for d in details if not d["passed"])

        return GateResult(
            passed=mean_score >= self.threshold,
            score=round(mean_score, 4),
            threshold=self.threshold,
            n_samples=len(test_suite),
            fail_count=fail_count,
            details=details,
        )

    def report(self, result: GateResult) -> None:
        """Print a human-readable gate report to stdout."""
        print("\n" + "=" * 60)
        print("  HALLUCINATION GATE REPORT")
        print("=" * 60)
        print(f"  {result.summary_line}")
        print()

        for i, d in enumerate(result.details, 1):
            status = "PASS" if d["passed"] else "FAIL"
            print(f"  [{i:02d}] [{status}] score={d['score']:.4f}  Q: {d['question'][:60]}")
            if "error" in d:
                print(f"         ERROR: {d['error']}")

        print()
        if result.passed:
            print("  Gate passed. Hallucination score is within acceptable range.")
        else:
            print(f"  Gate FAILED. Mean score {result.score:.4f} < threshold {result.threshold}.")
            print(f"  {result.fail_count} sample(s) fell below threshold.")
        print("=" * 60 + "\n")

    def enforce(self, result: GateResult) -> None:
        """
        Exit the process with code 1 if the gate failed.

        This is the line that makes CI fail the build. Call it after `report`.
        """
        if not result.passed and self.exit_on_fail:
            print("Exiting with code 1 — hallucination gate failed.", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Mock scorer (no API keys required for dry runs)
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_scorer(question: str, answer: str, context: list[str]) -> float:
        """
        Deterministic mock faithfulness score.

        Returns the Jaccard similarity between answer tokens and context tokens.
        Real deployment must replace this with the CombinedHallucinationScorer
        from Chapter 2.
        """
        ctx_tokens = set(
            w.lower() for passage in context for w in passage.split()
        )
        ans_tokens = set(answer.lower().split())
        if not ctx_tokens or not ans_tokens:
            return 0.0
        intersection = ctx_tokens & ans_tokens
        union = ctx_tokens | ans_tokens
        return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# 1b. GateConfig + HallucinationCIGate — the complete CI/CD gate (Listing 3.7)
# ---------------------------------------------------------------------------
#
# HallucinationCheckGate above (Listing 3.2) is the minimal merge-blocking check:
# one run, one threshold, one sys.exit(1). Section 3.7 assembles a more
# complete gate on top of that idea — configuration management (3.7.1), a
# re-run policy that absorbs judge-score flakiness (3.7.2), and baseline
# drift detection for model/provider changes (3.7.3). GateConfig and
# HallucinationCIGate below implement that assembled gate for real.

if _PYDANTIC_AVAILABLE:

    class GateConfig(_PydanticBaseModel):
        """Configuration for the hallucination CI/CD gate (Listing 3.7)."""

        threshold: float = 0.05          # Max acceptable hallucination rate
        sample_size: int = 100           # Examples per eval run
        concurrency: int = 20            # Parallel API calls
        max_reruns: int = 2              # Reruns before definitive fail
        rerun_pass_count: int = 2        # Passes out of max_reruns+1 needed
        baseline_drift_threshold: float = 0.02  # Alert if rate shifts > 2pp
        baseline_file: str = ".hallucination_baseline.json"
        golden_dataset: str = "data/golden_dataset_ci.jsonl"
        model: str = "gpt-4o-mini"
        judge_model: str = "gpt-4o-mini"
        report_path: str = "reports/gate_report.json"

else:

    @dataclass
    class GateConfig:  # type: ignore[no-redef]
        """
        Configuration for the hallucination CI/CD gate (Listing 3.7).

        Dataclass fallback used when pydantic isn't installed, so this
        module stays importable without the full ML/validation stack.
        Field names and defaults match the pydantic version exactly.
        """

        threshold: float = 0.05          # Max acceptable hallucination rate
        sample_size: int = 100           # Examples per eval run
        concurrency: int = 20            # Parallel API calls
        max_reruns: int = 2              # Reruns before definitive fail
        rerun_pass_count: int = 2        # Passes out of max_reruns+1 needed
        baseline_drift_threshold: float = 0.02  # Alert if rate shifts > 2pp
        baseline_file: str = ".hallucination_baseline.json"
        golden_dataset: str = "data/golden_dataset_ci.jsonl"
        model: str = "gpt-4o-mini"
        judge_model: str = "gpt-4o-mini"
        report_path: str = "reports/gate_report.json"


@dataclass
class RerunAttempt:
    """One evaluation attempt within the re-run policy (section 3.7.2)."""
    attempt_number: int
    passed: bool
    hallucination_rate: float
    n_samples: int


@dataclass
class CIGateResult:
    """Outcome of a full CI/CD gate run, including re-run and drift data."""
    passed: bool
    final_rate: float
    threshold: float
    attempts: list[RerunAttempt] = field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    baseline_drift_detected: bool = False
    baseline_drift_amount: Optional[float] = None
    warning: Optional[str] = None

    @property
    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        drift = (
            f" drift={self.baseline_drift_amount:+.4f}"
            if self.baseline_drift_detected else ""
        )
        return (
            f"[{status}] rate={self.final_rate:.4f} threshold={self.threshold} "
            f"attempts={len(self.attempts)} passes={self.pass_count}/"
            f"{len(self.attempts)}{drift}"
        )


class HallucinationCIGate:
    """
    The complete CI/CD hallucination gate (Listing 3.7).

    Three layers on top of the single-shot HallucinationCheckGate check:

    1. Configuration management (section 3.7.1) — every tunable lives on
       GateConfig instead of being scattered across call sites.
    2. The re-run policy (section 3.7.2) — LLM-as-judge evaluation isn't
       deterministic, so a single failing run doesn't block the merge on
       its own. If the first run fails, the gate re-runs on a fresh sample
       slice up to `config.max_reruns` additional times, and requires
       `config.rerun_pass_count` passes out of the attempts made before
       blocking. This is the same flaky-test-handling pattern CI systems
       apply to non-deterministic test suites.
    3. Baseline drift detection (section 3.7.3) — the gate persists the
       hallucination rate from the most recent full run to `baseline_file`
       and flags a shift bigger than `baseline_drift_threshold` as a
       baseline-drift warning (model/provider change) instead of quietly
       passing or failing for the wrong reason.

    Parameters
    ----------
    config : GateConfig
        Gate configuration (see GateConfig for field descriptions).
    scorer : callable, optional
        A function (question, answer, context) -> float returning a
        hallucination-rate score in [0, 1], where 0 is fully grounded and
        1 is fully hallucinated. If None, uses a deterministic mock scorer
        (1 - Jaccard token overlap with context) suitable for dry runs.
        Real deployment replaces this with the CombinedHallucinationScorer
        from chapter 2.
    """

    def __init__(
        self,
        config: GateConfig,
        scorer: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.scorer = scorer or self._mock_hallucination_rate_scorer

    # ------------------------------------------------------------------
    # Mock scorer (hallucination-rate direction, not faithfulness)
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_hallucination_rate_scorer(
        question: str, answer: str, context: list[str]
    ) -> float:
        """
        Deterministic mock hallucination-rate score: 0.0 = fully grounded,
        1.0 = fully hallucinated. This is 1 minus the Jaccard token-overlap
        scorer HallucinationCheckGate uses for faithfulness, because GateConfig's
        `threshold` is a rate ceiling ("max acceptable hallucination rate"),
        the opposite direction from HallucinationCheckGate's faithfulness floor.
        """
        ctx_tokens = set(
            w.lower() for passage in context for w in passage.split()
        )
        ans_tokens = set(answer.lower().split())
        if not ctx_tokens or not ans_tokens:
            return 1.0
        intersection = ctx_tokens & ans_tokens
        union = ctx_tokens | ans_tokens
        faithfulness = len(intersection) / len(union)
        return round(1.0 - faithfulness, 4)

    # ------------------------------------------------------------------
    # Re-run policy (section 3.7.2)
    # ------------------------------------------------------------------

    def _fresh_sample(self, golden_dataset: list[dict], attempt_index: int) -> list[dict]:
        """
        Return a fresh slice of `sample_size` examples for this attempt.

        Rotates the starting offset per attempt so a re-run doesn't just
        re-score the exact same examples that already failed — matching
        section 3.7.2's "run again with a fresh sample" rule — while
        staying deterministic (no random.sample) so the same attempt
        number always draws the same slice, per section 3.1.2's
        deterministic-sampling rule.
        """
        n = len(golden_dataset)
        if n == 0:
            return []
        size = min(self.config.sample_size, n)
        start = (attempt_index * size) % n
        if start + size <= n:
            return golden_dataset[start:start + size]
        return golden_dataset[start:] + golden_dataset[: (start + size) - n]

    def _run_once(self, golden_dataset: list[dict], attempt_index: int) -> float:
        """
        Score one fresh sample slice and return the mean hallucination rate.

        Scoring runs on a thread pool sized by `config.concurrency`
        (section 3.1.2): each `self.scorer(...)` call is dispatched to a
        worker thread, so `concurrency` real API calls (or, for the mock
        scorer, real function calls) are in flight at once instead of
        running the sample strictly one at a time. Threads, not asyncio,
        because `scorer` is a plain synchronous callable — swapping in
        `asyncio.gather` under a `Semaphore(config.concurrency)` is the
        right move if you replace the scorer with an async client.
        """
        sample = self._fresh_sample(golden_dataset, attempt_index)
        if not sample:
            return 1.0
        max_workers = max(1, min(self.config.concurrency, len(sample)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            scores = list(pool.map(
                lambda item: self.scorer(item["question"], item["answer"], item["context"]),
                sample,
            ))
        return round(sum(scores) / len(scores), 4)

    def run_with_rerun_policy(self, golden_dataset: list[dict]) -> CIGateResult:
        """
        Run the gate under the re-run policy from section 3.7.2.

        Runs up to `max_reruns + 1` total attempts. Stops early once the
        pass/fail outcome is mathematically decided (either enough passes
        to clear `rerun_pass_count`, or too few attempts remain to reach
        it), so a clean first-run pass doesn't burn extra API calls.
        """
        total_attempts = self.config.max_reruns + 1
        attempts: list[RerunAttempt] = []
        pass_count = 0
        fail_count = 0

        for i in range(total_attempts):
            rate = self._run_once(golden_dataset, i)
            passed = rate <= self.config.threshold
            attempts.append(RerunAttempt(
                attempt_number=i + 1,
                passed=passed,
                hallucination_rate=rate,
                n_samples=min(self.config.sample_size, len(golden_dataset)),
            ))
            pass_count += 1 if passed else 0
            fail_count += 0 if passed else 1

            if pass_count >= self.config.rerun_pass_count:
                break
            remaining = total_attempts - (i + 1)
            if pass_count + remaining < self.config.rerun_pass_count:
                break

        final_passed = pass_count >= self.config.rerun_pass_count
        final_rate = attempts[-1].hallucination_rate if attempts else 1.0

        warning = None
        if final_passed and fail_count > 0:
            warning = (
                f"Gate passed on a re-run ({pass_count}/{len(attempts)} attempts "
                f"passed) — at least one evaluation run was flaky."
            )
        elif not final_passed and pass_count > 0:
            warning = (
                f"Gate failed despite {pass_count}/{len(attempts)} passing "
                f"attempts — fewer than rerun_pass_count={self.config.rerun_pass_count} passed."
            )

        drift_detected, drift_amount = self._check_baseline_drift(final_rate)

        return CIGateResult(
            passed=final_passed,
            final_rate=final_rate,
            threshold=self.config.threshold,
            attempts=attempts,
            pass_count=pass_count,
            fail_count=fail_count,
            baseline_drift_detected=drift_detected,
            baseline_drift_amount=drift_amount,
            warning=warning,
        )

    # ------------------------------------------------------------------
    # Baseline drift detection (section 3.7.3)
    # ------------------------------------------------------------------

    def _load_baseline(self) -> Optional[dict]:
        path = Path(self.config.baseline_file)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _check_baseline_drift(self, current_rate: float) -> tuple[bool, Optional[float]]:
        """
        Compare `current_rate` against the stored baseline hallucination
        rate. A shift larger than `baseline_drift_threshold` in either
        direction is flagged as drift — most often a model/provider
        version change rather than a real prompt regression.
        """
        baseline = self._load_baseline()
        if baseline is None or "hallucination_rate" not in baseline:
            return False, None
        drift = round(current_rate - baseline["hallucination_rate"], 4)
        drift_detected = abs(drift) > self.config.baseline_drift_threshold
        return drift_detected, drift

    def update_baseline(self, current_rate: float, model: Optional[str] = None) -> None:
        """Persist the current run's hallucination rate as the new baseline."""
        path = Path(self.config.baseline_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "hallucination_rate": round(current_rate, 4),
            "model": model or self.config.model,
        }, indent=2))

    # ------------------------------------------------------------------
    # Reporting and enforcement
    # ------------------------------------------------------------------

    def report(self, result: CIGateResult) -> None:
        """Print a human-readable gate report to stdout."""
        print("\n" + "=" * 60)
        print("  CI/CD HALLUCINATION GATE REPORT (Listing 3.7)")
        print("=" * 60)
        print(f"  {result.summary_line}")
        print()
        for a in result.attempts:
            status = "PASS" if a.passed else "FAIL"
            print(
                f"  Attempt {a.attempt_number}: [{status}] "
                f"rate={a.hallucination_rate:.4f} n={a.n_samples}"
            )
        if result.baseline_drift_detected:
            print(
                f"\n  BASELINE DRIFT WARNING: rate shifted "
                f"{result.baseline_drift_amount:+.4f} vs. stored baseline "
                f"(threshold {self.config.baseline_drift_threshold})."
            )
            print("  Recalibrate the threshold; do not treat this as a PR rejection.")
        if result.warning:
            print(f"\n  NOTE: {result.warning}")
        print()
        if result.passed:
            print("  Gate PASSED.")
        else:
            print(
                f"  Gate FAILED. {result.pass_count}/{len(result.attempts)} "
                f"attempts passed threshold {result.threshold}."
            )
        print("=" * 60 + "\n")

    def write_report(self, result: CIGateResult) -> None:
        """Write the gate report as a CI artifact (section 3.7.4)."""
        path = Path(self.config.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "passed": result.passed,
            "final_rate": result.final_rate,
            "threshold": result.threshold,
            "pass_count": result.pass_count,
            "fail_count": result.fail_count,
            "attempts": [
                {
                    "attempt_number": a.attempt_number,
                    "passed": a.passed,
                    "hallucination_rate": a.hallucination_rate,
                    "n_samples": a.n_samples,
                }
                for a in result.attempts
            ],
            "baseline_drift_detected": result.baseline_drift_detected,
            "baseline_drift_amount": result.baseline_drift_amount,
            "warning": result.warning,
        }, indent=2))

    def enforce(self, result: CIGateResult, exit_on_fail: bool = True) -> None:
        """Exit the process with code 1 if the gate failed."""
        if not result.passed and exit_on_fail:
            print("Exiting with code 1 — hallucination CI gate failed.", file=sys.stderr)
            sys.exit(1)


# ---------------------------------------------------------------------------
# 1c. Prompt version tracking (section 3.2.1)
# ---------------------------------------------------------------------------

def hash_prompt(prompt: str) -> str:
    """Short, stable content hash used to tag a system prompt version."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def evaluate_prompt_version(
    system_prompt: str,
    golden_examples: list[dict],
    threshold: float = 0.5,
    completion_fn: Optional[Callable] = None,
    scorer: Optional[Callable] = None,
) -> dict:
    """
    Evaluate a system prompt version against the golden dataset (section 3.2.1).

    Returns a summary result suitable for version comparison: the prompt's
    content hash, the mean hallucination score across `golden_examples`,
    and whether this version passes `threshold`.

    Parameters
    ----------
    system_prompt : str
        The prompt version under test. May contain `{name}`-style
        placeholders filled from each example's `vars` dict.
    golden_examples : list[dict]
        Each dict needs a `"query"` key, and may include `"vars"` (format
        substitutions for `system_prompt`) and `"context"` (list[str]
        passed to `scorer`).
    threshold : float
        Max acceptable mean hallucination score. Default 0.5.
    completion_fn : callable, optional
        Function (question, system_prompt, model, temperature) -> str.
        If None, uses the OpenAI API when `OPENAI_API_KEY` is set, else
        `_mock_completion` — the same fallback `SelfConsistencyChecker` uses.
    scorer : callable, optional
        Function (question, answer, context) -> float, hallucination-rate
        direction (0 = grounded, 1 = hallucinated). If None, uses
        `HallucinationCIGate._mock_hallucination_rate_scorer`. Replace with
        the `CombinedHallucinationScorer` from chapter 2 for production use.
    """
    import os
    if completion_fn is not None:
        fn = completion_fn
    elif _OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        fn = _openai_completion
    else:
        fn = _mock_completion
    score_fn = scorer or HallucinationCIGate._mock_hallucination_rate_scorer

    scores = []
    for ex in golden_examples:
        prompt_text = system_prompt.format(**ex.get("vars", {}))
        answer = fn(ex["query"], prompt_text, "gpt-4o-mini", 0.0)
        scores.append(score_fn(ex["query"], answer, ex.get("context", [])))

    mean_score = round(sum(scores) / len(scores), 4) if scores else 1.0
    return {
        "prompt_hash": hash_prompt(system_prompt),
        "mean_hallucination_score": mean_score,
        "threshold": threshold,
        "passed": mean_score <= threshold,
        "n_examples": len(golden_examples),
    }


# ---------------------------------------------------------------------------
# 1d. Before/after prompt comparison (Listing 3.3, section 3.2.2)
# ---------------------------------------------------------------------------

PROMPT_BEFORE = """You are a helpful customer service assistant for Acme Corp.
Answer the user's question about our return policy."""

PROMPT_AFTER = """You are a customer service assistant for Acme Corp.
Answer ONLY based on the policy document provided below.
If the user asks about something not covered in the document,
say exactly: "I don't have information on that in our current policy.
Please contact support@acme.com for clarification."

Do not infer, extrapolate, or guess. If you are uncertain, use the fallback.

POLICY DOCUMENT:
{policy_text}
"""

EXAMPLES = [
    {
        "query": "Can I return a product after 60 days if it's defective?",
        "policy_text": "Acme Corp accepts returns within 30 days of purchase "
                       "for any reason. Defective items may be returned within "
                       "90 days with proof of defect.",
        "expected_before_behavior": "May hallucinate a 60-day defective-item "
                                    "policy that does not exist.",
        "expected_after_behavior": "States the 90-day defective-item window "
                                   "explicitly from the policy.",
    },
    {
        "query": "What is your holiday return extension policy?",
        "policy_text": "Acme Corp accepts returns within 30 days of purchase "
                       "for any reason. Defective items may be returned within "
                       "90 days with proof of defect.",
        "expected_before_behavior": "May fabricate a holiday extension policy.",
        "expected_after_behavior": "Uses the fallback: states it has no "
                                   "information on a holiday extension and "
                                   "points the user to support@acme.com.",
    },
    {
        "query": "Do you offer free return shipping on orders over $50?",
        "policy_text": "Acme Corp accepts returns within 30 days of purchase "
                       "for any reason. Defective items may be returned within "
                       "90 days with proof of defect.",
        "expected_before_behavior": "May invent a free-shipping threshold that "
                                    "sounds plausible for a retail return policy.",
        "expected_after_behavior": "Uses the fallback, since shipping cost is "
                                   "not covered in the policy document.",
    },
]


# ---------------------------------------------------------------------------
# 2. SelfConsistencyChecker
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    """Result of a self-consistency check on a single question."""
    question: str
    responses: list[str]
    majority_answer: str
    agreement_rate: float   # fraction of responses matching majority
    consistent: bool        # True if agreement_rate >= min_agreement
    min_agreement: float


class SelfConsistencyChecker:
    """
    Sample N completions for a question and measure how often they agree.

    Self-consistency is a simple but effective hallucination signal: if a model
    produces contradictory answers when asked the same question N times, the
    response is unreliable. High variance responses should be flagged before
    being returned to users or used in downstream pipelines.

    Parameters
    ----------
    n_samples : int
        Number of completions to sample per question. Default 5.
    min_agreement : float
        Minimum fraction of responses that must match the majority answer.
        Default 0.6.
    model : str
        Model to call for completions. Default "gpt-4o-mini".
    temperature : float
        Sampling temperature. Must be > 0 to get varied samples. Default 0.8.
    """

    def __init__(
        self,
        n_samples: int = 5,
        min_agreement: float = 0.6,
        model: str = "gpt-4o-mini",
        temperature: float = 0.8,
    ) -> None:
        self.n_samples = n_samples
        self.min_agreement = min_agreement
        self.model = model
        self.temperature = temperature

    def check(
        self,
        question: str,
        system_prompt: str = "Answer concisely.",
        completion_fn: Optional[Callable] = None,
    ) -> ConsistencyResult:
        """
        Sample N completions for a question and compute agreement.

        Parameters
        ----------
        question : str
            The question to evaluate.
        system_prompt : str
            System prompt prepended to each call.
        completion_fn : callable, optional
            Function (question, system_prompt, model, temperature) -> str.
            If None, uses the OpenAI API if available, or the mock generator.

        Returns
        -------
        ConsistencyResult with responses, majority answer, and agreement rate.
        """
        fn = completion_fn or self._get_completion_fn()
        responses = []
        for _ in range(self.n_samples):
            try:
                resp = fn(question, system_prompt, self.model, self.temperature)
                responses.append(resp.strip())
            except Exception as exc:
                responses.append(f"[ERROR: {exc}]")

        majority, agreement_rate = self._compute_majority(responses)
        return ConsistencyResult(
            question=question,
            responses=responses,
            majority_answer=majority,
            agreement_rate=round(agreement_rate, 4),
            consistent=agreement_rate >= self.min_agreement,
            min_agreement=self.min_agreement,
        )

    @staticmethod
    def _compute_majority(responses: list[str]) -> tuple[str, float]:
        """Return the most common response and its frequency."""
        # Normalize for comparison: lowercase, strip punctuation
        normalized = [r.lower().strip(".!?") for r in responses]
        counts: dict[str, int] = {}
        for r in normalized:
            counts[r] = counts.get(r, 0) + 1
        majority_norm = max(counts, key=lambda k: counts[k])
        majority_raw = next(r for r, n in zip(responses, normalized) if n == majority_norm)
        agreement_rate = counts[majority_norm] / len(responses)
        return majority_raw, agreement_rate

    def _get_completion_fn(self) -> Callable:
        import os
        if _OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            return _openai_completion
        return _mock_completion


def _openai_completion(
    question: str, system_prompt: str, model: str, temperature: float
) -> str:
    """Call the OpenAI Chat Completions API for a single response."""
    client = _openai_module.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        max_tokens=256,
    )
    return response.choices[0].message.content


def _mock_completion(
    question: str, system_prompt: str, model: str, temperature: float
) -> str:
    """Return a randomly varied mock answer (no API key required)."""
    candidates = [
        "Paris is the capital of France.",
        "The capital of France is Paris.",
        "France's capital city is Paris.",
        "Paris serves as the capital of France.",
        "Lyon is the capital of France.",   # intentional outlier
    ]
    return random.choice(candidates)


def _mock_prompt_completion(
    question: str, system_prompt: str, model: str, temperature: float
) -> str:
    """
    Mock completion tuned for the prompt-version demo (`evaluate_prompt_version`,
    section 3.2.1). Returns a plausible-sounding fabricated answer for the
    loose PROMPT_BEFORE style, and a grounded, policy-quoting answer (or the
    exact fallback string) for the strict PROMPT_AFTER style, so the demo
    shows a real scoring difference between the two prompt versions instead
    of two prompts scoring identically against an unrelated mock answer.
    """
    strict = "ONLY based on the policy document" in system_prompt
    if "60 days" in question:
        if strict:
            return "Defective items may be returned within 90 days with proof of defect."
        return "Yes, you can return it within 60 days since it's defective."
    if "holiday" in question:
        if strict:
            return ("I don't have information on that in our current policy. "
                     "Please contact support@acme.com for clarification.")
        return "We extend the holiday return window by an extra 30 days."
    if "free return shipping" in question:
        if strict:
            return ("I don't have information on that in our current policy. "
                     "Please contact support@acme.com for clarification.")
        return "Yes, orders over $50 get free return shipping."
    return "I don't have information on that."


# ---------------------------------------------------------------------------
# 3. ClaimDecompositionPipeline
# ---------------------------------------------------------------------------

@dataclass
class DecomposedClaim:
    """A single atomic claim extracted from a compound response."""
    claim: str
    supported: Optional[bool] = None   # True/False/None (unverified)
    evidence: Optional[str] = None


@dataclass
class DecompositionResult:
    """Result of decomposing a compound answer into atomic claims."""
    original_answer: str
    claims: list[DecomposedClaim]
    support_rate: Optional[float] = None   # fraction of claims marked supported

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    @property
    def n_supported(self) -> int:
        return sum(1 for c in self.claims if c.supported is True)


class ClaimDecompositionPipeline:
    """
    Decompose compound LLM answers into atomic claims for granular scoring.

    A single faithfulness score on a multi-sentence answer hides per-claim
    accuracy. A response that is 80% correct can still contain one catastrophic
    hallucination. Decomposition makes each claim independently auditable.

    The pipeline:
    1. Split the answer into candidate sentences.
    2. Use an LLM (or rule-based fallback) to rewrite each sentence as a
       standalone, independently verifiable claim.
    3. Optionally score each claim against provided context.

    Parameters
    ----------
    model : str
        Model to use for claim extraction. Default "gpt-4o-mini".
    score_claims : bool
        If True, run a simple keyword-overlap check to mark each claim as
        supported or unsupported by the context. Default True.
    """

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a claim extraction assistant. Given a passage, extract each
        factual claim as a single, self-contained, atomic sentence.
        Return ONLY a JSON array of strings, one string per claim.
        Example output: ["Paris is the capital of France.", "Paris is on the Seine river."]
    """).strip()

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        score_claims: bool = True,
    ) -> None:
        self.model = model
        self.score_claims = score_claims

    def decompose(
        self,
        answer: str,
        context: list[str] | None = None,
        extraction_fn: Optional[Callable] = None,
    ) -> DecompositionResult:
        """
        Decompose an answer into atomic claims.

        Parameters
        ----------
        answer : str
            The model-generated response to decompose.
        context : list[str], optional
            Retrieved context to score claims against.
        extraction_fn : callable, optional
            Function (answer) -> list[str] that returns atomic claims.
            If None, uses OpenAI if available, else a sentence-split fallback.

        Returns
        -------
        DecompositionResult with one DecomposedClaim per atomic claim.
        """
        fn = extraction_fn or self._get_extraction_fn()
        raw_claims = fn(answer)

        claims = [DecomposedClaim(claim=c) for c in raw_claims]

        if self.score_claims and context:
            claims = self._score_against_context(claims, context)

        supported = [c for c in claims if c.supported is True]
        support_rate = (
            len(supported) / len(claims) if claims else None
        )

        return DecompositionResult(
            original_answer=answer,
            claims=claims,
            support_rate=round(support_rate, 4) if support_rate is not None else None,
        )

    @staticmethod
    def _score_against_context(
        claims: list[DecomposedClaim], context: list[str]
    ) -> list[DecomposedClaim]:
        """
        Mark each claim as supported/unsupported using token overlap.

        This is a keyword heuristic, not a semantic NLI check. Replace with
        an NLI model (e.g. cross-encoder/nli-deberta-v3-base) for production.
        """
        context_tokens = set(
            w.lower() for passage in context for w in passage.split()
        )
        for claim in claims:
            claim_tokens = set(claim.claim.lower().split())
            overlap = len(context_tokens & claim_tokens)
            # Threshold: at least 40% of claim tokens must appear in context
            claim.supported = overlap / max(len(claim_tokens), 1) >= 0.4
            if claim.supported:
                claim.evidence = "Keyword overlap with retrieved context."
            else:
                claim.evidence = "No supporting tokens found in context."
        return claims

    def _get_extraction_fn(self) -> Callable:
        import os
        if _OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            return self._openai_extract
        return self._sentence_split_extract

    def _openai_extract(self, answer: str) -> list[str]:
        client = _openai_module.OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": answer},
            ],
            temperature=0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        # Handle both {"claims": [...]} and direct [...]
        if isinstance(parsed, list):
            return parsed
        for key in ("claims", "sentences", "facts"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        return list(parsed.values())[0] if parsed else []

    @staticmethod
    def _sentence_split_extract(answer: str) -> list[str]:
        """Fallback: split on sentence boundaries."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# 4. ShadowTrafficHarness
# ---------------------------------------------------------------------------

@dataclass
class ShadowComparison:
    """Comparison of production vs candidate model for one request."""
    request_id: str
    question: str
    production_answer: str
    candidate_answer: str
    production_score: float
    candidate_score: float
    candidate_wins: bool
    delta: float


@dataclass
class ShadowReport:
    """Aggregate results from a shadow traffic run."""
    total: int
    candidate_wins: int
    production_wins: int
    ties: int
    mean_production_score: float
    mean_candidate_score: float
    mean_delta: float
    recommendation: str
    # Machine-readable companion to `recommendation` above: just the
    # lowercase decision token ("promote" / "hold" / "block"), with no
    # score embedded in the string. `recommendation` is built for a human
    # reading a shadow-traffic report; `recommendation_code` is built for
    # a caller like ch11's PRHardeningGate that branches on the verdict
    # programmatically (`shadow.get("recommendation_code", "unknown") in
    # ("promote", "hold")`) and would never match against the descriptive
    # string.
    recommendation_code: str
    comparisons: list[ShadowComparison] = field(default_factory=list)


class ShadowTrafficHarness:
    """
    Replay production traffic against a candidate model and compare quality.

    Shadow traffic testing catches failure modes that lab evaluation misses:
    the long tail of unusual, ambiguous, or adversarial inputs that real users
    generate but that do not appear in curated test sets.

    The harness:
    1. Accepts a batch of production request logs.
    2. Sends each request to both the production model and the candidate model.
    3. Scores each response with a faithfulness scorer.
    4. Returns a statistical comparison and a go/no-go recommendation.

    Parameters
    ----------
    production_fn : callable
        Function (question, context) -> str for the production model.
    candidate_fn : callable
        Function (question, context) -> str for the candidate model.
    scorer_fn : callable
        Function (question, answer, context) -> float.
    min_improvement : float
        Minimum mean score delta (candidate - production) required to
        recommend promotion. Default 0.02 (2 percentage points).
    max_regression : float
        Maximum allowed mean score drop (negative delta) before the harness
        recommends blocking the promotion. Default -0.05.
    """

    def __init__(
        self,
        production_fn: Optional[Callable] = None,
        candidate_fn: Optional[Callable] = None,
        scorer_fn: Optional[Callable] = None,
        min_improvement: float = 0.02,
        max_regression: float = -0.05,
    ) -> None:
        self.production_fn = production_fn or _mock_model_a
        self.candidate_fn = candidate_fn or _mock_model_b
        self.scorer_fn = scorer_fn or _jaccard_score
        self.min_improvement = min_improvement
        self.max_regression = max_regression

    def run(
        self,
        traffic_logs: list[dict],
        sample_pct: float = 1.0,
    ) -> ShadowReport:
        """
        Run shadow comparison on a batch of production logs.

        Parameters
        ----------
        traffic_logs : list[dict]
            Each dict must have: 'question' (str), 'context' (list[str]).
            Optional: 'request_id' (str).
        sample_pct : float
            Fraction of traffic to shadow. Default 1.0 (100%).

        Returns
        -------
        ShadowReport with per-request comparisons and aggregate stats.
        """
        sample_size = max(1, int(len(traffic_logs) * sample_pct))
        sampled = random.sample(traffic_logs, sample_size)

        comparisons: list[ShadowComparison] = []
        for i, log in enumerate(sampled):
            rid = log.get("request_id", f"req-{i+1:04d}")
            question = log["question"]
            context = log.get("context", [])

            prod_answer = self.production_fn(question, context)
            cand_answer = self.candidate_fn(question, context)

            prod_score = self.scorer_fn(question, prod_answer, context)
            cand_score = self.scorer_fn(question, cand_answer, context)

            delta = cand_score - prod_score
            comparisons.append(ShadowComparison(
                request_id=rid,
                question=question,
                production_answer=prod_answer,
                candidate_answer=cand_answer,
                production_score=round(prod_score, 4),
                candidate_score=round(cand_score, 4),
                candidate_wins=delta > 0,
                delta=round(delta, 4),
            ))

        mean_prod = sum(c.production_score for c in comparisons) / len(comparisons)
        mean_cand = sum(c.candidate_score for c in comparisons) / len(comparisons)
        mean_delta = mean_cand - mean_prod

        wins = sum(1 for c in comparisons if c.candidate_wins)
        losses = sum(1 for c in comparisons if not c.candidate_wins and c.delta != 0)
        ties = len(comparisons) - wins - losses

        recommendation_code = self._classify(mean_delta)
        recommendation = self._recommend(mean_delta, recommendation_code)

        return ShadowReport(
            total=len(comparisons),
            candidate_wins=wins,
            production_wins=losses,
            ties=ties,
            mean_production_score=round(mean_prod, 4),
            mean_candidate_score=round(mean_cand, 4),
            mean_delta=round(mean_delta, 4),
            recommendation=recommendation,
            recommendation_code=recommendation_code,
            comparisons=comparisons,
        )

    def _classify(self, mean_delta: float) -> str:
        """
        Return the lowercase decision token a caller can branch on:
        "promote", "hold", or "block". This is `recommendation_code` on
        the returned ShadowReport.
        """
        if mean_delta >= self.min_improvement:
            return "promote"
        if mean_delta <= self.max_regression:
            return "block"
        return "hold"

    def _recommend(self, mean_delta: float, code: str) -> str:
        """Build the human-readable `recommendation` string for `code`."""
        if code == "promote":
            return f"PROMOTE — candidate improves mean score by {mean_delta:+.4f}."
        if code == "block":
            return f"BLOCK — candidate regresses mean score by {mean_delta:+.4f}."
        return f"HOLD — delta {mean_delta:+.4f} is within tolerance. Extend shadow window."

    def print_report(self, report: ShadowReport) -> None:
        print("\n" + "=" * 60)
        print("  SHADOW TRAFFIC REPORT")
        print("=" * 60)
        print(f"  Total requests  : {report.total}")
        print(f"  Candidate wins  : {report.candidate_wins}")
        print(f"  Production wins : {report.production_wins}")
        print(f"  Ties            : {report.ties}")
        print(f"  Mean prod score : {report.mean_production_score:.4f}")
        print(f"  Mean cand score : {report.mean_candidate_score:.4f}")
        print(f"  Mean delta      : {report.mean_delta:+.4f}")
        print()
        print(f"  RECOMMENDATION: {report.recommendation}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_model_a(question: str, context: list[str]) -> str:
    """Simulates the production model (slightly noisier)."""
    words = question.split()
    return f"Based on available information: {' '.join(words[:5])} answer here."


def _mock_model_b(question: str, context: list[str]) -> str:
    """Simulates the candidate model (uses context tokens)."""
    ctx_words = " ".join(context[:1])[:80] if context else "the provided context"
    return f"According to context: {ctx_words}."


def _jaccard_score(question: str, answer: str, context: list[str]) -> float:
    """Token-level Jaccard similarity between answer and context."""
    ctx_tokens = set(w.lower() for p in context for w in p.split())
    ans_tokens = set(answer.lower().split())
    if not ctx_tokens or not ans_tokens:
        return 0.0
    return len(ctx_tokens & ans_tokens) / len(ctx_tokens | ans_tokens)


# ---------------------------------------------------------------------------
# 4b. Canary window sizing (section 3.6)
# ---------------------------------------------------------------------------

def canary_window_sizing(
    baseline_rate: float,
    detectable_shift: float,
    total_qph: int,
    canary_fraction: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Compute the canary window duration required to reach statistical power
    for detecting a hallucination rate regression (section 3.6).

    Uses the standard two-proportion z-test sample-size formula: the
    number of canary requests needed per arm to detect a shift from
    `baseline_rate` to `baseline_rate + detectable_shift` at significance
    `alpha` with `power`, then converts that sample size to a wall-clock
    window given the canary's share of total traffic.

    Args:
        baseline_rate: Current production hallucination rate (e.g. 0.02 for 2%).
        detectable_shift: Minimum shift to detect (e.g. 0.01 for 1 pp shift).
        total_qph: Total queries per hour through the system.
        canary_fraction: Fraction of traffic routed to canary (e.g. 0.05).
        alpha: Significance level for the two-proportion z-test.
        power: Desired statistical power (0.80 = 80%).

    Returns:
        dict with required sample size (`n_required`), the canary's
        queries-per-hour (`canary_qph`), and the window duration in hours
        (`window_hours`) needed to accumulate that sample.
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "canary_window_sizing requires scipy (pip install scipy>=1.11.0)"
        )

    p1 = baseline_rate
    p2 = baseline_rate + detectable_shift
    p_pooled = (p1 + p2) / 2

    z_alpha = _scipy_norm.ppf(1 - alpha / 2)
    z_power = _scipy_norm.ppf(power)

    numerator = (
        z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled))
        + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2
    n_required = math.ceil(numerator / denominator)

    canary_qph = total_qph * canary_fraction
    window_hours = n_required / canary_qph if canary_qph > 0 else float("inf")

    return {
        "n_required": n_required,
        "canary_qph": round(canary_qph, 2),
        "window_hours": round(window_hours, 2),
    }


# ---------------------------------------------------------------------------
# 5. GitHub Actions YAML generator
# ---------------------------------------------------------------------------

GITHUB_ACTIONS_YAML = """\
# .github/workflows/hallucination-gate.yml
# Chapter 3 — Hardening LLM Systems in Production
# Hallucination CI gate: fails the build if mean faithfulness < threshold.
# Scope matches manuscript Listing 3.1 exactly: PR-to-main only, and only
# when a change could plausibly move the hallucination rate (src/, prompts/,
# or config/ changed) — not on every PR to every branch.

name: Hallucination Rate Gate

on:
  pull_request:
    branches: [main]
    paths:
      - "src/**"
      - "prompts/**"
      - "config/**"

env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  HALLUCINATION_THRESHOLD: "0.80"   # Fail build if mean score < 0.80
  TEST_SUITE_PATH: "tests/hallucination/test_suite.json"

jobs:
  hallucination-gate:
    name: Hallucination Rate Gate
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install evaluation dependencies
        run: |
          pip install --upgrade pip
          pip install deepeval==0.21.71 ragas==0.2.15 \\
                      scikit-learn>=1.3.0 scipy>=1.11.0 openai>=1.0.0

      - name: Run hallucination gate
        run: |
          python ch03_scripts.py --gate \\
            --threshold ${{ env.HALLUCINATION_THRESHOLD }} \\
            --suite ${{ env.TEST_SUITE_PATH }}
        # sys.exit(1) on failure causes this step to fail,
        # which blocks the PR from merging.

      - name: Upload gate report
        if: always()   # upload even on failure so engineers can debug
        uses: actions/upload-artifact@v4
        with:
          name: hallucination-gate-report
          path: gate_report.json
          retention-days: 30
"""


def print_github_actions_yaml() -> None:
    print("\n" + "=" * 60)
    print("  GITHUB ACTIONS YAML — Hallucination Gate")
    print("=" * 60)
    print(GITHUB_ACTIONS_YAML)


# ---------------------------------------------------------------------------
# Demo test suites
# ---------------------------------------------------------------------------

DEMO_TEST_SUITE = [
    {
        "request_id": "test-001",
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris, located on the Seine river.",
        "context": ["Paris is the capital of France.", "It is located on the Seine."],
    },
    {
        "request_id": "test-002",
        "question": "Who invented the telephone?",
        "answer": "The telephone was invented by Alexander Graham Bell in 1876.",
        "context": ["Alexander Graham Bell is credited with inventing the telephone.", "He received the first patent in 1876."],
    },
    {
        "request_id": "test-003",
        "question": "When did World War II end?",
        "answer": "World War II ended in 1945 with the surrender of Germany and Japan.",
        "context": ["WWII ended in 1945.", "Germany surrendered in May 1945, Japan in September 1945."],
    },
    {
        "request_id": "test-004",
        "question": "What is the speed of light?",
        "answer": "The speed of light is approximately 300,000 km/s in a vacuum.",
        "context": ["The speed of light in a vacuum is approximately 299,792 km/s."],
    },
    {
        "request_id": "test-005",
        "question": "Who wrote Hamlet?",
        "answer": "Hamlet was written by Charles Dickens in the 17th century.",
        "context": ["Hamlet is a tragedy written by William Shakespeare, believed to have been written around 1600-1601."],
    },
]

DEMO_TRAFFIC_LOGS = [
    {
        "request_id": f"prod-{i+1:03d}",
        "question": item["question"],
        "context": item["context"],
    }
    for i, item in enumerate(DEMO_TEST_SUITE)
]


# ---------------------------------------------------------------------------
# CI/CD gate demo (Listing 3.7): config management, re-run policy, drift
# ---------------------------------------------------------------------------

def _demo_ci_gate() -> None:
    """
    Demo for the complete CI/CD gate (Listing 3.7): configuration
    management via GateConfig, the re-run policy (section 3.7.2), and
    baseline drift detection (section 3.7.3), run end to end against the
    same DEMO_TEST_SUITE used by the other demos.
    """
    tmp_dir = tempfile.mkdtemp(prefix="ch03_gate_demo_")
    baseline_path = str(Path(tmp_dir) / "baseline.json")

    config = GateConfig(
        threshold=0.66,
        sample_size=3,
        max_reruns=2,
        rerun_pass_count=2,
        baseline_drift_threshold=0.02,
        baseline_file=baseline_path,
        report_path=str(Path(tmp_dir) / "gate_report.json"),
    )
    gate = HallucinationCIGate(config)

    print(
        f"  GateConfig: threshold={config.threshold} sample_size={config.sample_size} "
        f"max_reruns={config.max_reruns} rerun_pass_count={config.rerun_pass_count}"
    )

    # First run: no baseline on disk yet, so no drift check fires.
    result = gate.run_with_rerun_policy(DEMO_TEST_SUITE)
    gate.report(result)
    gate.write_report(result)
    gate.update_baseline(result.final_rate)
    print(f"  Baseline written: {result.final_rate:.4f} -> {baseline_path}")

    # Second run: simulate a model/provider change that shifts the rate.
    # Same data, but a scorer biased +0.35 higher — this should trip the
    # baseline-drift warning from section 3.7.3 rather than read as a
    # plain prompt regression.
    def _drifted_scorer(question, answer, context):
        base = HallucinationCIGate._mock_hallucination_rate_scorer(question, answer, context)
        return round(min(base + 0.35, 1.0), 4)

    drift_gate = HallucinationCIGate(config, scorer=_drifted_scorer)
    drift_result = drift_gate.run_with_rerun_policy(DEMO_TEST_SUITE)
    print("\n  --- Second run: simulated model/provider change ---")
    drift_gate.report(drift_result)

    gate.enforce(result, exit_on_fail=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 3 companion — CI/CD hallucination containment demos"
    )
    parser.add_argument("--gate", action="store_true", help="Run CI gate demo (Listing 3.2)")
    parser.add_argument(
        "--ci-gate", action="store_true",
        help="Run full CI/CD gate demo: config, re-run policy, baseline drift (Listing 3.7)",
    )
    parser.add_argument("--consistency", action="store_true", help="Run self-consistency demo")
    parser.add_argument("--claims", action="store_true", help="Run claim decomposition demo")
    parser.add_argument("--shadow", action="store_true", help="Run shadow traffic demo")
    parser.add_argument(
        "--prompt-version", action="store_true",
        help="Run prompt version tracking + before/after demo (section 3.2.1-3.2.2, Listing 3.3)",
    )
    parser.add_argument(
        "--canary-sizing", action="store_true",
        help="Run canary window power-analysis demo (section 3.6)",
    )
    parser.add_argument("--gen-yaml", action="store_true", help="Print GitHub Actions YAML")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--suite", type=str, default=None)
    args = parser.parse_args()

    # If no flags given, run everything
    run_all = not any([
        args.gate, args.ci_gate, args.consistency, args.claims, args.shadow,
        args.prompt_version, args.canary_sizing, args.gen_yaml,
    ])

    print("\nChapter 3: Containing Hallucinations as a CI/CD-Blocking Metric")
    print("Hardening LLM Systems in Production — Companion Code")
    print("=" * 60)

    if args.gate or run_all:
        print("\n--- CI Gate Demo ---")
        gate = HallucinationCheckGate(threshold=args.threshold, exit_on_fail=False)

        suite = DEMO_TEST_SUITE
        if args.suite:
            with open(args.suite) as fh:
                suite = json.load(fh)

        result = gate.run(suite)
        gate.report(result)
        # Write JSON report for CI artifact upload
        with open("gate_report.json", "w") as fh:
            json.dump(
                {
                    "passed": result.passed,
                    "score": result.score,
                    "threshold": result.threshold,
                    "details": result.details,
                },
                fh, indent=2,
            )
        print("  gate_report.json written.")

    if args.ci_gate or run_all:
        print("\n--- Full CI/CD Gate Demo (Listing 3.7) ---")
        _demo_ci_gate()

    if args.consistency or run_all:
        print("\n--- Self-Consistency Demo ---")
        checker = SelfConsistencyChecker(
            n_samples=5,
            min_agreement=0.6,
        )
        result = checker.check(
            "What is the capital of France?",
            completion_fn=_mock_completion,
        )
        print(f"  Question         : {result.question}")
        print(f"  Majority answer  : {result.majority_answer}")
        print(f"  Agreement rate   : {result.agreement_rate:.2%}")
        print(f"  Consistent (>={result.min_agreement:.0%}) : {result.consistent}")
        print(f"  All responses    :")
        for i, r in enumerate(result.responses, 1):
            print(f"    [{i}] {r}")

    if args.claims or run_all:
        print("\n--- Claim Decomposition Demo ---")
        pipeline = ClaimDecompositionPipeline(score_claims=True)
        answer = (
            "The Eiffel Tower was built by Gustave Eiffel in 1887. "
            "It stands 330 meters tall and is located in Berlin."
        )
        context = [
            "The Eiffel Tower was constructed between 1887 and 1889.",
            "It was built by Gustave Eiffel's company.",
            "The tower is located on the Champ de Mars in Paris.",
            "Its height is approximately 330 metres.",
        ]
        result = pipeline.decompose(answer, context=context)
        print(f"  Original answer  : {result.original_answer}")
        print(f"  Claims extracted : {result.n_claims}")
        print(f"  Support rate     : {result.support_rate:.2%}")
        for i, claim in enumerate(result.claims, 1):
            status = "SUPPORTED" if claim.supported else "UNSUPPORTED"
            print(f"    [{i}] [{status}] {claim.claim}")

    if args.shadow or run_all:
        print("\n--- Shadow Traffic Demo ---")
        harness = ShadowTrafficHarness(
            production_fn=_mock_model_a,
            candidate_fn=_mock_model_b,
            scorer_fn=_jaccard_score,
        )
        report = harness.run(DEMO_TRAFFIC_LOGS, sample_pct=1.0)
        harness.print_report(report)

    if args.prompt_version or run_all:
        print("\n--- Prompt Version Tracking Demo (section 3.2.1, Listing 3.3) ---")
        golden_examples = [
            {
                "query": ex["query"],
                "vars": {"policy_text": ex["policy_text"]},
                "context": [ex["policy_text"]],
            }
            for ex in EXAMPLES
        ]
        before_result = evaluate_prompt_version(
            PROMPT_BEFORE, golden_examples, completion_fn=_mock_prompt_completion,
        )
        after_result = evaluate_prompt_version(
            PROMPT_AFTER, golden_examples, completion_fn=_mock_prompt_completion,
        )
        print(f"  PROMPT_BEFORE  hash={before_result['prompt_hash']}  "
              f"mean_score={before_result['mean_hallucination_score']:.4f}  "
              f"passed={before_result['passed']}")
        print(f"  PROMPT_AFTER   hash={after_result['prompt_hash']}  "
              f"mean_score={after_result['mean_hallucination_score']:.4f}  "
              f"passed={after_result['passed']}")
        for ex in EXAMPLES:
            print(f"    Q: {ex['query']}")
            print(f"      before: {ex['expected_before_behavior']}")
            print(f"      after : {ex['expected_after_behavior']}")

    if args.canary_sizing or run_all:
        print("\n--- Canary Window Sizing Demo (section 3.6) ---")
        if _SCIPY_AVAILABLE:
            small_shift = canary_window_sizing(
                baseline_rate=0.05, detectable_shift=0.03,
                total_qph=5000, canary_fraction=0.02,
            )
            print(f"  small-shift scenario: n_required={small_shift['n_required']} "
                  f"canary_qph={small_shift['canary_qph']} "
                  f"window_hours={small_shift['window_hours']}")
            large_shift = canary_window_sizing(
                baseline_rate=0.02, detectable_shift=0.01,
                total_qph=1000, canary_fraction=0.05,
            )
            print(f"  3000-5000 request scenario: n_required={large_shift['n_required']} "
                  f"canary_qph={large_shift['canary_qph']} "
                  f"window_hours={large_shift['window_hours']}")
        else:
            print("  scipy not installed — skipping (pip install scipy>=1.11.0)")

    if args.gen_yaml or run_all:
        print_github_actions_yaml()


if __name__ == "__main__":
    main()
