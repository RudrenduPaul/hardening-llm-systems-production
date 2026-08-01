"""
Chapter 3: Detecting Hallucinations Before Your Users Do
Hardening LLM Systems in Production — Companion Code
Author: Rudrendu Paul | https://orcid.org/0009-0008-0141-4690

Implements:
  - HallucinationMetric wrapper around deepeval
  - RAGAS faithfulness scoring pipeline
  - CombinedHallucinationScorer (ensemble of both)
  - Inter-rater reliability via Cohen's kappa
  - Statistical power analysis for sample size planning

Requirements:
    deepeval==0.21.7
    ragas==0.1.21
    scikit-learn>=1.3.0,<2.0
    scipy>=1.11.0,<2.0
    datasets>=2.14.0,<3.0
    openai>=1.0.0,<2.0
    langchain>=0.1.0,<1.0
    langchain-openai>=0.0.5,<1.0

Usage:
    python ch03_scripts.py
    python ch03_scripts.py --demo faithfulness
    python ch03_scripts.py --demo kappa
    python ch03_scripts.py --demo power
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

# Suppress verbose library warnings in demo output
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Type stubs so the module is importable without the heavy ML stack installed
# (the stubs are replaced at runtime if the real packages are present)
# ---------------------------------------------------------------------------

import os as _os

try:
    from deepeval import evaluate
    from deepeval.metrics import HallucinationMetric as _DeepEvalHallucinationMetric
    from deepeval.test_case import LLMTestCase
    # deepeval requires OPENAI_API_KEY at runtime; treat missing key as unavailable
    _DEEPEVAL_AVAILABLE = bool(_os.environ.get("OPENAI_API_KEY"))
except ImportError:
    _DEEPEVAL_AVAILABLE = False

try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    _RAGAS_AVAILABLE = bool(_os.environ.get("OPENAI_API_KEY"))
except ImportError:
    _RAGAS_AVAILABLE = False

try:
    from sklearn.metrics import cohen_kappa_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. deepeval HallucinationMetric wrapper
# ---------------------------------------------------------------------------

@dataclass
class HallucinationResult:
    """Result from a single deepeval hallucination check."""
    input: str
    actual_output: str
    context: list[str]
    score: float          # 0.0 = hallucinated, 1.0 = faithful
    passed: bool
    reason: str
    metric_name: str = "HallucinationMetric"


class HallucinationMetric:
    """
    Thin wrapper around deepeval's HallucinationMetric.

    Normalizes the interface so downstream code does not depend directly
    on deepeval's internal API, making it easier to swap detectors.

    Parameters
    ----------
    threshold : float
        Minimum faithfulness score required to pass. Default 0.5.
    model : str
        Judge model passed to deepeval. Default "gpt-4o".
    """

    def __init__(self, threshold: float = 0.5, model: str = "gpt-4o") -> None:
        self.threshold = threshold
        self.model = model
        self._metric = None
        if _DEEPEVAL_AVAILABLE:
            self._metric = _DeepEvalHallucinationMetric(
                threshold=threshold,
                model=model,
                include_reason=True,
            )

    def score(
        self,
        input_text: str,
        actual_output: str,
        context: list[str],
    ) -> HallucinationResult:
        """
        Score a single response for hallucination.

        Parameters
        ----------
        input_text : str
            The original user question or prompt.
        actual_output : str
            The model's response to evaluate.
        context : list[str]
            Retrieved context chunks (RAG) or ground-truth passages.

        Returns
        -------
        HallucinationResult with score, pass/fail, and explanation.
        """
        if not _DEEPEVAL_AVAILABLE:
            return self._mock_result(input_text, actual_output, context)

        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            context=context,
        )
        self._metric.measure(test_case)
        return HallucinationResult(
            input=input_text,
            actual_output=actual_output,
            context=context,
            score=self._metric.score,
            passed=self._metric.is_successful(),
            reason=self._metric.reason or "No reason provided",
        )

    def batch_score(
        self,
        samples: list[dict],
    ) -> list[HallucinationResult]:
        """
        Score a list of samples.

        Each sample dict must contain keys: 'input', 'actual_output', 'context'.
        'context' should be a list of strings.
        """
        results = []
        for s in samples:
            results.append(
                self.score(
                    input_text=s["input"],
                    actual_output=s["actual_output"],
                    context=s["context"],
                )
            )
        return results

    @staticmethod
    def _mock_result(
        input_text: str, actual_output: str, context: list[str]
    ) -> HallucinationResult:
        """Return a deterministic mock result when deepeval is not installed."""
        # Simple heuristic: if the output contains words from the context it is
        # more likely faithful. This is NOT a substitute for the real metric.
        context_words = set(
            word.lower()
            for passage in context
            for word in passage.split()
        )
        output_words = set(actual_output.lower().split())
        overlap = len(context_words & output_words)
        score = min(1.0, overlap / max(len(output_words), 1))
        return HallucinationResult(
            input=input_text,
            actual_output=actual_output,
            context=context,
            score=round(score, 3),
            passed=score >= 0.5,
            reason="[MOCK — install deepeval==0.21.7 for real scoring]",
        )

    def summary_stats(self, results: list[HallucinationResult]) -> dict:
        """Compute aggregate statistics from a batch of results."""
        scores = [r.score for r in results]
        passes = [r.passed for r in results]
        return {
            "n": len(scores),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "pass_rate": round(sum(passes) / len(passes), 4) if passes else 0.0,
            "fail_count": sum(1 for p in passes if not p),
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# 2. RAGAS faithfulness scoring pipeline
# ---------------------------------------------------------------------------

@dataclass
class RAGASFaithfulnessResult:
    """Result from RAGAS faithfulness evaluation."""
    faithfulness_score: float
    answer_relevancy_score: Optional[float]
    context_recall_score: Optional[float]
    n_samples: int


class RAGASPipeline:
    """
    Evaluate hallucination risk in RAG pipelines using RAGAS faithfulness.

    RAGAS (Retrieval Augmented Generation Assessment) decomposes each
    response into atomic claims, then checks each claim against the
    retrieved context. The faithfulness score is the fraction of claims
    supported by context.

    Parameters
    ----------
    metrics : list[str]
        Which RAGAS metrics to run. Options: "faithfulness", "answer_relevancy",
        "context_recall". Default: ["faithfulness"].
    """

    SUPPORTED_METRICS = ["faithfulness", "answer_relevancy", "context_recall"]

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or ["faithfulness"]
        self._validate_metrics()

    def _validate_metrics(self) -> None:
        unknown = [m for m in self.metrics if m not in self.SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"Unknown RAGAS metrics: {unknown}. "
                f"Supported: {self.SUPPORTED_METRICS}"
            )

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> RAGASFaithfulnessResult:
        """
        Run RAGAS evaluation on a batch of QA pairs.

        Parameters
        ----------
        questions : list[str]
            User queries.
        answers : list[str]
            Model-generated answers.
        contexts : list[list[str]]
            Retrieved context chunks per question.
        ground_truths : list[str], optional
            Reference answers. Required for context_recall.

        Returns
        -------
        RAGASFaithfulnessResult with per-metric scores.
        """
        if not _RAGAS_AVAILABLE:
            return self._mock_result(len(questions))

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths is not None:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        metric_objects = []
        if "faithfulness" in self.metrics:
            metric_objects.append(faithfulness)
        if "answer_relevancy" in self.metrics:
            metric_objects.append(answer_relevancy)
        if "context_recall" in self.metrics and ground_truths is not None:
            metric_objects.append(context_recall)

        result = ragas_evaluate(dataset, metrics=metric_objects)
        result_dict = result.to_pandas().mean().to_dict()

        return RAGASFaithfulnessResult(
            faithfulness_score=round(result_dict.get("faithfulness", 0.0), 4),
            answer_relevancy_score=round(result_dict.get("answer_relevancy", 0.0), 4)
            if "answer_relevancy" in self.metrics else None,
            context_recall_score=round(result_dict.get("context_recall", 0.0), 4)
            if "context_recall" in self.metrics else None,
            n_samples=len(questions),
        )

    @staticmethod
    def _mock_result(n: int) -> RAGASFaithfulnessResult:
        """Return a mock result when RAGAS is not installed."""
        return RAGASFaithfulnessResult(
            faithfulness_score=0.82,
            answer_relevancy_score=0.75,
            context_recall_score=None,
            n_samples=n,
        )


# ---------------------------------------------------------------------------
# 3. CombinedHallucinationScorer — ensemble of deepeval + RAGAS
# ---------------------------------------------------------------------------

@dataclass
class CombinedScore:
    """Ensemble hallucination score combining deepeval and RAGAS."""
    deepeval_score: float
    ragas_faithfulness: float
    combined_score: float
    passed: bool
    threshold: float
    weights: tuple[float, float]   # (deepeval_weight, ragas_weight)
    details: dict = field(default_factory=dict)


class CombinedHallucinationScorer:
    """
    Weighted ensemble of deepeval HallucinationMetric and RAGAS faithfulness.

    Using two independent metrics reduces single-point-of-failure risk: each
    metric uses a different decomposition strategy, so their agreement gives
    stronger evidence of faithfulness than either alone.

    Parameters
    ----------
    deepeval_weight : float
        Weight assigned to the deepeval score (0–1). Default 0.6.
    ragas_weight : float
        Weight assigned to RAGAS faithfulness (0–1). Default 0.4.
        deepeval_weight + ragas_weight must equal 1.0.
    threshold : float
        Minimum combined score to pass. Default 0.7.
    deepeval_model : str
        Judge model for deepeval. Default "gpt-4o".
    """

    def __init__(
        self,
        deepeval_weight: float = 0.6,
        ragas_weight: float = 0.4,
        threshold: float = 0.7,
        deepeval_model: str = "gpt-4o",
    ) -> None:
        if not math.isclose(deepeval_weight + ragas_weight, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"deepeval_weight ({deepeval_weight}) + ragas_weight ({ragas_weight}) "
                f"must sum to 1.0"
            )
        self.weights = (deepeval_weight, ragas_weight)
        self.threshold = threshold
        self._de_metric = HallucinationMetric(
            threshold=threshold, model=deepeval_model
        )
        self._ragas = RAGASPipeline(metrics=["faithfulness"])

    def score(
        self,
        question: str,
        answer: str,
        context: list[str],
        ground_truth: str | None = None,
    ) -> CombinedScore:
        """
        Score a single QA pair with the ensemble.

        Parameters
        ----------
        question : str
            User question.
        answer : str
            Model-generated answer.
        context : list[str]
            Retrieved context chunks.
        ground_truth : str, optional
            Reference answer (improves RAGAS scoring when available).

        Returns
        -------
        CombinedScore with individual and ensemble scores.
        """
        de_result = self._de_metric.score(
            input_text=question,
            actual_output=answer,
            context=context,
        )

        gts = [ground_truth] if ground_truth else None
        ragas_result = self._ragas.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[context],
            ground_truths=gts,
        )

        combined = (
            self.weights[0] * de_result.score
            + self.weights[1] * ragas_result.faithfulness_score
        )
        combined = round(combined, 4)

        return CombinedScore(
            deepeval_score=de_result.score,
            ragas_faithfulness=ragas_result.faithfulness_score,
            combined_score=combined,
            passed=combined >= self.threshold,
            threshold=self.threshold,
            weights=self.weights,
            details={
                "deepeval_reason": de_result.reason,
                "deepeval_passed": de_result.passed,
            },
        )


# ---------------------------------------------------------------------------
# 4. Cohen's kappa for inter-rater reliability
# ---------------------------------------------------------------------------

def compute_cohens_kappa(
    rater_a: list[int],
    rater_b: list[int],
    weights: str | None = None,
) -> dict:
    """
    Compute Cohen's kappa between two human (or model) raters.

    Use this to validate that your automated metric agrees with human
    judgements before trusting it as a CI gate.

    Parameters
    ----------
    rater_a, rater_b : list[int]
        Binary (0/1) or ordinal labels from each rater. Must be the same length.
    weights : None | "linear" | "quadratic"
        Weighting scheme for ordinal labels. Use None for binary.

    Returns
    -------
    dict with 'kappa', 'interpretation', and 'n' keys.

    Interpretation guide
    --------------------
    kappa < 0.00  : Less than chance agreement
    0.00–0.20     : Slight
    0.21–0.40     : Fair
    0.41–0.60     : Moderate
    0.61–0.80     : Substantial
    0.81–1.00     : Almost perfect
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(
            f"Rater lists must be the same length. "
            f"Got {len(rater_a)} and {len(rater_b)}."
        )

    if not _SKLEARN_AVAILABLE:
        # Compute kappa manually (binary case only)
        kappa = _manual_kappa(rater_a, rater_b)
    else:
        kappa = float(cohen_kappa_score(rater_a, rater_b, weights=weights))

    return {
        "kappa": round(kappa, 4),
        "n": len(rater_a),
        "interpretation": _interpret_kappa(kappa),
        "acceptable_for_ci_gate": kappa >= 0.61,
    }


def _manual_kappa(a: list[int], b: list[int]) -> float:
    """Simple manual Cohen's kappa for binary labels (no sklearn required)."""
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n

    # Expected agreement by chance
    a_pos = sum(a) / n
    b_pos = sum(b) / n
    pe = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _interpret_kappa(k: float) -> str:
    if k < 0:
        return "Less than chance"
    if k < 0.21:
        return "Slight"
    if k < 0.41:
        return "Fair"
    if k < 0.61:
        return "Moderate"
    if k < 0.81:
        return "Substantial"
    return "Almost perfect"


# ---------------------------------------------------------------------------
# 5. Power analysis for sample size planning
# ---------------------------------------------------------------------------

def compute_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Estimate the number of samples needed to detect a change in hallucination rate.

    Use this before launching an evaluation study to ensure your sample size
    gives you enough statistical power to catch real regressions.

    Parameters
    ----------
    baseline_rate : float
        Current hallucination rate (e.g. 0.15 for 15%).
    minimum_detectable_effect : float
        Smallest absolute change you need to detect (e.g. 0.05 for 5 percentage points).
    alpha : float
        Type I error rate (false positive probability). Default 0.05.
    power : float
        Statistical power (1 - Type II error rate). Default 0.80.

    Returns
    -------
    dict with 'n_per_group', 'n_total', 'effect_size', and interpretation.

    Notes
    -----
    Uses the two-proportion z-test formula. For LLM evaluations, treat the
    baseline model and candidate model as two independent groups.
    """
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect

    # Clip probabilities to valid range
    p1 = max(0.001, min(0.999, p1))
    p2 = max(0.001, min(0.999, p2))

    if _SCIPY_AVAILABLE:
        z_alpha = float(_scipy_stats.norm.ppf(1 - alpha / 2))
        z_beta = float(_scipy_stats.norm.ppf(power))
    else:
        # Approximations for common values
        _z_table = {0.05: 1.96, 0.01: 2.576, 0.10: 1.645}
        _z_power = {0.80: 0.842, 0.90: 1.282, 0.95: 1.645}
        z_alpha = _z_table.get(alpha, 1.96)
        z_beta = _z_power.get(power, 0.842)

    # Cohen's h effect size for proportions
    h = 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))

    # Sample size per group
    n_per_group = math.ceil((z_alpha + z_beta) ** 2 / h ** 2)

    return {
        "baseline_rate": p1,
        "alternative_rate": p2,
        "minimum_detectable_effect": minimum_detectable_effect,
        "alpha": alpha,
        "power": power,
        "cohens_h": round(abs(h), 4),
        "n_per_group": n_per_group,
        "n_total": n_per_group * 2,
        "interpretation": (
            f"You need {n_per_group} samples per model to detect a "
            f"{minimum_detectable_effect:.0%} shift in hallucination rate "
            f"with {power:.0%} power at alpha={alpha}."
        ),
    }


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

DEMO_SAMPLES = [
    {
        "input": "What is the capital of France?",
        "actual_output": "The capital of France is Paris. It is located in the north of France on the Seine river.",
        "context": [
            "Paris is the capital and largest city of France.",
            "The city is situated on the Seine river in northern France.",
        ],
        "ground_truth": "Paris",
    },
    {
        "input": "When was the Eiffel Tower built?",
        "actual_output": "The Eiffel Tower was built in 1887 and completed in 1889 for the World's Fair.",
        "context": [
            "The Eiffel Tower was constructed between 1887 and 1889.",
            "It was built as the entrance arch for the 1889 World's Fair in Paris.",
        ],
        "ground_truth": "Construction began in 1887 and was completed in 1889.",
    },
    {
        "input": "What is the population of Tokyo?",
        "actual_output": "Tokyo has a population of approximately 14 million people in the city proper.",
        "context": [
            "Tokyo Metropolis has a population of about 14 million in its 23 special wards.",
            "The greater Tokyo area has a population of approximately 37 million.",
        ],
        "ground_truth": "About 14 million in the city proper, 37 million in the greater area.",
    },
    {
        "input": "Who wrote the Iliad?",
        "actual_output": "The Iliad was written by Shakespeare in the 14th century.",
        "context": [
            "The Iliad is an ancient Greek epic poem attributed to Homer.",
            "It is believed to have been composed in the 8th century BC.",
        ],
        "ground_truth": "Homer is the attributed author; composed circa 8th century BC.",
    },
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_faithfulness_demo() -> None:
    print("\n== deepeval HallucinationMetric Demo ==\n")
    metric = HallucinationMetric(threshold=0.5, model="gpt-4o")

    results = metric.batch_score(
        [{"input": s["input"], "actual_output": s["actual_output"], "context": s["context"]}
         for s in DEMO_SAMPLES]
    )
    for i, r in enumerate(results):
        status = "PASS" if r.passed else "FAIL"
        print(f"  Sample {i+1} [{status}] score={r.score:.3f}")
        print(f"    Q: {r.input[:60]}")
        print(f"    A: {r.actual_output[:80]}")
        print(f"    Reason: {r.reason[:100]}")
        print()

    stats = metric.summary_stats(results)
    print(f"  Batch stats: {json.dumps(stats, indent=4)}")

    print("\n== RAGAS Faithfulness Demo ==\n")
    pipeline = RAGASPipeline(metrics=["faithfulness"])
    ragas_result = pipeline.evaluate(
        questions=[s["input"] for s in DEMO_SAMPLES],
        answers=[s["actual_output"] for s in DEMO_SAMPLES],
        contexts=[s["context"] for s in DEMO_SAMPLES],
        ground_truths=[s["ground_truth"] for s in DEMO_SAMPLES],
    )
    print(f"  Faithfulness score     : {ragas_result.faithfulness_score:.4f}")
    print(f"  Samples evaluated      : {ragas_result.n_samples}")

    print("\n== CombinedHallucinationScorer Demo ==\n")
    scorer = CombinedHallucinationScorer(
        deepeval_weight=0.6,
        ragas_weight=0.4,
        threshold=0.7,
    )
    sample = DEMO_SAMPLES[3]  # The Shakespeare hallucination
    combined = scorer.score(
        question=sample["input"],
        answer=sample["actual_output"],
        context=sample["context"],
        ground_truth=sample["ground_truth"],
    )
    print(f"  deepeval score     : {combined.deepeval_score:.4f}")
    print(f"  RAGAS faithfulness : {combined.ragas_faithfulness:.4f}")
    print(f"  Combined score     : {combined.combined_score:.4f}")
    print(f"  Passed (>={combined.threshold})   : {combined.passed}")


def run_kappa_demo() -> None:
    print("\n== Cohen's Kappa Demo ==\n")

    # Simulated labels from two human raters on 20 LLM responses
    # 0 = faithful, 1 = hallucinated
    human_a = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
    human_b = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1]

    kappa = compute_cohens_kappa(human_a, human_b)
    print(f"  Kappa              : {kappa['kappa']:.4f}")
    print(f"  Interpretation     : {kappa['interpretation']}")
    print(f"  Acceptable for CI  : {kappa['acceptable_for_ci_gate']}")
    print(f"  N samples          : {kappa['n']}")

    # Model-vs-human agreement
    model_scores = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    model_kappa = compute_cohens_kappa(human_a, model_scores)
    print(f"\n  Model-vs-Human kappa : {model_kappa['kappa']:.4f}  [{model_kappa['interpretation']}]")
    if model_kappa["acceptable_for_ci_gate"]:
        print("  => Model agreement is substantial enough to use as a CI gate.")
    else:
        print("  => Model agreement is too low. Increase human validation before deploying as CI gate.")


def run_power_demo() -> None:
    print("\n== Power Analysis Demo ==\n")

    scenarios = [
        (0.15, 0.05),   # 15% baseline, detect 5pp improvement
        (0.15, 0.10),   # 15% baseline, detect 10pp improvement
        (0.30, 0.05),   # 30% baseline, detect 5pp improvement
    ]

    for baseline, mde in scenarios:
        result = compute_sample_size(
            baseline_rate=baseline,
            minimum_detectable_effect=mde,
        )
        print(f"  Baseline={baseline:.0%} | MDE={mde:.0%}")
        print(f"    n per group : {result['n_per_group']}")
        print(f"    n total     : {result['n_total']}")
        print(f"    Cohen's h   : {result['cohens_h']}")
        print(f"    {result['interpretation']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 3 companion — hallucination detection demos"
    )
    parser.add_argument(
        "--demo",
        choices=["faithfulness", "kappa", "power", "all"],
        default="all",
        help="Which demo to run. Default: all",
    )
    args = parser.parse_args()

    print("\nChapter 3: Detecting Hallucinations Before Your Users Do")
    print("Hardening LLM Systems in Production — Companion Code")
    print("=" * 60)

    if not _DEEPEVAL_AVAILABLE:
        print("\n[NOTE] deepeval not installed — using mock scorer.")
        print("       pip install deepeval==0.21.7\n")
    if not _RAGAS_AVAILABLE:
        print("[NOTE] ragas not installed — using mock scorer.")
        print("       pip install ragas==0.1.21\n")

    if args.demo in ("faithfulness", "all"):
        run_faithfulness_demo()
    if args.demo in ("kappa", "all"):
        run_kappa_demo()
    if args.demo in ("power", "all"):
        run_power_demo()


if __name__ == "__main__":
    main()
