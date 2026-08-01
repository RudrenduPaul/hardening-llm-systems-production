"""
Chapter 2: Detecting Hallucinations Before Your Users Do
Hardening LLM Systems in Production — Companion Code
Author: Rudrendu Paul | https://orcid.org/0009-0008-0141-4690

Implements:
  - Chain-of-thought drift detection (embedding similarity)
  - BERTScore computation
  - HallucinationMetric wrapper around deepeval
  - RAGAS faithfulness scoring pipeline
  - CombinedHallucinationScorer (ensemble of both)
  - LLM-as-judge sycophancy calibration
  - Inter-rater reliability via Cohen's kappa
  - Statistical power analysis for sample size planning
  - HallucinationGate: CI/CD gate-compatible scorer wrapper

Requirements:
    deepeval==0.21.7
    ragas==0.1.21
    scikit-learn>=1.3.0,<2.0
    scipy>=1.11.0,<2.0
    numpy>=1.24.0,<2.0
    bert-score>=0.3.13,<1.0
    datasets>=2.14.0,<3.0
    openai>=1.0.0,<2.0
    langchain>=0.1.0,<1.0
    langchain-openai>=0.0.5,<1.0

Usage:
    python ch02_scripts.py
    python ch02_scripts.py --demo faithfulness
    python ch02_scripts.py --demo kappa
    python ch02_scripts.py --demo power
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter
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

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import openai as _openai
    # Embeddings calls need OPENAI_API_KEY too; treat missing key as unavailable
    _OPENAI_EMBEDDINGS_AVAILABLE = bool(_os.environ.get("OPENAI_API_KEY"))
except ImportError:
    _OPENAI_EMBEDDINGS_AVAILABLE = False

try:
    from bert_score import score as _bert_score_fn
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Chain-of-thought drift detection (section 2.2.5)
# ---------------------------------------------------------------------------

def detect_cot_drift(
    final_reasoning_step: str,
    conclusion: str,
    drift_threshold: float = 0.82,
) -> dict:
    """
    Compare the semantic content of the final CoT step against the
    model's conclusion. A similarity below the threshold signals drift.

    Args:
        final_reasoning_step: The last step in the model's chain of thought.
        conclusion: The model's final answer or decision.
        drift_threshold: Cosine similarity below which we flag drift.
                         The 0.82 default is derived from the authors'
                         production deployments across contract-analysis
                         and legal-research tasks; it is not sourced from
                         a published benchmark. Run the power analysis in
                         section 2.7 against your domain's CoT pairs to
                         calibrate your own threshold before using it as
                         a gate.

    Returns:
        dict with similarity score, drift flag, and both inputs.
    """
    if not (_NUMPY_AVAILABLE and _OPENAI_EMBEDDINGS_AVAILABLE):
        return _mock_cot_drift(final_reasoning_step, conclusion, drift_threshold)

    client = _openai.OpenAI()
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[final_reasoning_step, conclusion],
    )
    vec_reasoning = np.array(resp.data[0].embedding)
    vec_conclusion = np.array(resp.data[1].embedding)

    similarity = float(
        np.dot(vec_reasoning, vec_conclusion)
        / (np.linalg.norm(vec_reasoning) * np.linalg.norm(vec_conclusion))
    )

    return {
        "similarity": round(similarity, 4),
        "drift_detected": similarity < drift_threshold,
        "drift_threshold": drift_threshold,
        "final_reasoning_step": final_reasoning_step,
        "conclusion": conclusion,
    }


def _mock_cot_drift(
    final_reasoning_step: str, conclusion: str, drift_threshold: float
) -> dict:
    """Deterministic mock CoT drift check when numpy/openai (or
    OPENAI_API_KEY) are not available. Uses word-overlap similarity in
    place of embedding cosine similarity. NOT a substitute for the real
    embedding-based comparison — install numpy and openai, and set
    OPENAI_API_KEY, for real scoring."""
    reasoning_words = set(final_reasoning_step.lower().split())
    conclusion_words = set(conclusion.lower().split())
    union = reasoning_words | conclusion_words
    similarity = (
        len(reasoning_words & conclusion_words) / len(union) if union else 0.0
    )
    return {
        "similarity": round(similarity, 4),
        "drift_detected": similarity < drift_threshold,
        "drift_threshold": drift_threshold,
        "final_reasoning_step": final_reasoning_step,
        "conclusion": conclusion,
        "reason": "[MOCK — install numpy and openai, and set OPENAI_API_KEY, "
                  "for real embedding-based scoring]",
    }


# ---------------------------------------------------------------------------
# 2. BERTScore computation (section 2.4.1)
# ---------------------------------------------------------------------------

def compute_bertscore(candidates: list[str], references: list[str]) -> list[dict]:
    """
    Compute BERTScore for a list of candidate-reference pairs.

    Args:
        candidates: Generated outputs to evaluate.
        references: Ground-truth reference texts.

    Returns:
        List of dicts with precision, recall, and F1 per pair.
    """
    if not _BERTSCORE_AVAILABLE:
        return _mock_bertscore(candidates, references)

    P, R, F1 = _bert_score_fn(
        candidates,
        references,
        lang="en",
        model_type="roberta-large",
        verbose=False,
    )
    return [
        {
            "candidate": c,
            "reference": r,
            "precision": round(p.item(), 4),
            "recall": round(rc.item(), 4),
            "f1": round(f.item(), 4),
        }
        for c, r, p, rc, f in zip(candidates, references, P, R, F1)
    ]


def _mock_bertscore(candidates: list[str], references: list[str]) -> list[dict]:
    """Deterministic mock BERTScore (word-overlap Jaccard) when bert-score
    is not installed. NOT a substitute for the real contextual-embedding
    metric — see section 2.4.1 for why token overlap misses the factual
    inversions BERTScore itself also misses."""
    results = []
    for c, r in zip(candidates, references):
        c_words = set(c.lower().split())
        r_words = set(r.lower().split())
        union = c_words | r_words
        jaccard = len(c_words & r_words) / len(union) if union else 0.0
        results.append({
            "candidate": c,
            "reference": r,
            "precision": round(jaccard, 4),
            "recall": round(jaccard, 4),
            "f1": round(jaccard, 4),
        })
    return results


# ---------------------------------------------------------------------------
# 3. deepeval HallucinationMetric wrapper (section 2.5.1)
# ---------------------------------------------------------------------------

@dataclass
class HallucinationResult:
    """Result from a single deepeval hallucination check."""
    input: str
    actual_output: str
    context: list[str]
    score: float          # 0.0 = faithful, 1.0 = hallucinated (higher = more hallucinated)
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
        Maximum hallucination score allowed to pass (score is 0.0 =
        faithful, 1.0 = hallucinated, so lower is better). Default 0.5.
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

    def _mock_result(
        self, input_text: str, actual_output: str, context: list[str]
    ) -> HallucinationResult:
        """Return a deterministic mock result when deepeval is not installed."""
        # Simple heuristic: the less the output's vocabulary is grounded in
        # the context, the more likely it is hallucinated. Score follows the
        # real deepeval polarity: 0.0 = faithful, 1.0 = hallucinated (higher
        # is worse). This is NOT a substitute for the real metric.
        context_words = set(
            word.lower()
            for passage in context
            for word in passage.split()
        )
        output_words = set(actual_output.lower().split())
        overlap = len(context_words & output_words)
        grounded_ratio = min(1.0, overlap / max(len(output_words), 1))
        score = round(1.0 - grounded_ratio, 3)
        return HallucinationResult(
            input=input_text,
            actual_output=actual_output,
            context=context,
            score=score,
            passed=score <= self.threshold,
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


def score_hallucination_deepeval(
    user_input: str,
    actual_output: str,
    context_documents: list[str],
    threshold: float = 0.7,
) -> dict:
    """
    Score a single LLM response for hallucination using deepeval.

    Listing 2.3 companion function — a thin functional wrapper around
    HallucinationMetric for readers following the book's inline code
    style; delegates to the same underlying deepeval integration.

    Args:
        user_input: The original user query.
        actual_output: The LLM's generated response to score.
        context_documents: The authoritative source documents retrieved
                           for this query (not the LLM's retrieved context,
                           but the ground-truth authoritative content).
        threshold: Maximum hallucination score allowed to pass.

    Returns:
        dict with keys: score, passed, reason
    """
    metric = HallucinationMetric(threshold=threshold, model="gpt-4o")
    result = metric.score(
        input_text=user_input,
        actual_output=actual_output,
        context=context_documents,
    )
    return {"score": result.score, "passed": result.passed, "reason": result.reason}


# ---------------------------------------------------------------------------
# 4. RAGAS faithfulness scoring pipeline (section 2.5.2)
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


def score_faithfulness_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
) -> dict:
    """
    Score a batch of LLM responses for faithfulness using RAGAS.

    Listing 2.4 companion function — a thin functional wrapper around
    RAGASPipeline for readers following the book's inline code style.

    RAGAS faithfulness decomposes each answer into atomic claims and
    checks each claim against the provided contexts. The score is the
    fraction of claims that are supported.

    Args:
        questions: List of original user queries.
        answers: List of LLM-generated answers to score.
        contexts: List of context document lists. Each entry is a list
                  of strings representing the grounding documents for
                  the corresponding question/answer pair.

    Returns:
        dict with 'faithfulness' (the dataset-mean faithfulness score)
        and 'n_samples' (the number of samples evaluated).
    """
    pipeline = RAGASPipeline(metrics=["faithfulness"])
    result = pipeline.evaluate(questions=questions, answers=answers, contexts=contexts)
    return {"faithfulness": result.faithfulness_score, "n_samples": result.n_samples}


# ---------------------------------------------------------------------------
# 5. CombinedHallucinationScorer — ensemble of deepeval + RAGAS (section 2.5.3)
# ---------------------------------------------------------------------------

@dataclass
class CombinedScore:
    """Ensemble hallucination score combining deepeval and RAGAS."""
    deepeval_score: float          # 0-1; higher = more hallucinated
    ragas_faithfulness: float      # 0-1; higher = more faithful (inverse polarity)
    combined_score: float          # 0-1; higher = higher hallucination risk
    passed: bool                   # True when combined_score <= threshold
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
        Maximum acceptable combined risk score to pass (0 = no risk,
        1 = maximum risk, so lower is better). Default 0.7.
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

        # deepeval's score is already "higher = more hallucinated." RAGAS
        # faithfulness is "higher = more faithful," so it must be inverted
        # to (1 - faithfulness) before combining, or the two signals point
        # in opposite directions and the sum is meaningless.
        combined = (
            self.weights[0] * de_result.score
            + self.weights[1] * (1 - ragas_result.faithfulness_score)
        )
        combined = round(combined, 4)

        return CombinedScore(
            deepeval_score=de_result.score,
            ragas_faithfulness=ragas_result.faithfulness_score,
            combined_score=combined,
            passed=combined <= self.threshold,
            threshold=self.threshold,
            weights=self.weights,
            details={
                "deepeval_reason": de_result.reason,
                "deepeval_passed": de_result.passed,
            },
        )


# ---------------------------------------------------------------------------
# 6. LLM-as-judge sycophancy calibration (section 2.5.4)
# ---------------------------------------------------------------------------

def build_sycophancy_pairs(
    correct_answers: list[str],
    incorrect_answers: list[str],
    context: str,
    question: str,
) -> list[dict]:
    """
    Build adversarial calibration pairs.

    Each pair contains:
      - tentative_correct: a correct answer with hedging language
      - confident_wrong: an incorrect answer stated confidently
    """
    pairs = []
    for correct, incorrect in zip(correct_answers, incorrect_answers):
        pairs.append(
            {
                "question": question,
                "context": context,
                "tentative_correct": f"I believe {correct.lower()}, though you may want to verify.",
                "confident_wrong": f"{incorrect} This is confirmed policy.",
            }
        )
    return pairs


def measure_sycophancy_rate(
    pairs: list[dict],
    judge: HallucinationMetric | None = None,
) -> dict:
    """
    Measure how often an LLM-as-judge favors a confidently wrong answer
    over a correctly hedged one.

    For each adversarial pair built by build_sycophancy_pairs(), scores
    both the tentative-but-correct and the confident-but-wrong variants
    with the same judge and counts how often the judge rates the
    confident wrong answer as LESS hallucinated (a lower score, since
    higher = more hallucinated) than the correct-but-hedged one. That
    inversion is a sycophancy event: the judge is rewarding confidence
    over accuracy.

    Args:
        pairs: Adversarial pairs from build_sycophancy_pairs(), each with
               'question', 'context', 'tentative_correct', and
               'confident_wrong' keys.
        judge: A HallucinationMetric instance to use as the judge.
               Defaults to a fresh HallucinationMetric(threshold=0.5).

    Returns:
        dict with 'sycophancy_rate', 'n_pairs', 'n_sycophantic', and
        'flagged_pairs' (indexes where the judge favored confidence over
        accuracy).
    """
    judge = judge or HallucinationMetric(threshold=0.5)
    n = len(pairs)
    flagged = []
    for i, pair in enumerate(pairs):
        context = [pair["context"]]
        correct_result = judge.score(
            input_text=pair["question"],
            actual_output=pair["tentative_correct"],
            context=context,
        )
        wrong_result = judge.score(
            input_text=pair["question"],
            actual_output=pair["confident_wrong"],
            context=context,
        )
        if wrong_result.score < correct_result.score:
            flagged.append(i)

    return {
        "sycophancy_rate": round(len(flagged) / n, 4) if n else 0.0,
        "n_pairs": n,
        "n_sycophantic": len(flagged),
        "flagged_pairs": flagged,
    }


# ---------------------------------------------------------------------------
# 7. Cohen's kappa for inter-rater reliability (section 2.6.3)
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


def compute_inter_rater_kappa(
    rater_a_labels: list[int],
    rater_b_labels: list[int],
    weights: str | None = None,
) -> dict:
    """
    Compute Cohen's kappa between two annotators, with a diagnostic
    breakdown of where they disagree.

    Listing 2.7 companion function — wraps compute_cohens_kappa() and
    adds the raw agreement rate plus a disagreement_types breakdown used
    to steer annotation-guideline revisions (see section 2.6.3).

    Args:
        rater_a_labels: Integer labels from annotator A.
                        For binary factuality: 0 = hallucinated, 1 = correct.
                        For graded: 0 = hallucinated, 1 = partial, 2 = correct.
        rater_b_labels: Integer labels from annotator B (same schema).
        weights: None for unweighted kappa (binary tasks),
                 'linear' or 'quadratic' for ordinal/graded labels.

    Returns:
        dict with kappa, n, interpretation, acceptable_for_ci_gate,
        raw_agreement, and disagreement_types (a count of
        (rater_a_label, rater_b_label) pairs where the two disagreed).
    """
    assert len(rater_a_labels) == len(rater_b_labels), (
        "Both raters must label the same number of examples."
    )

    result = compute_cohens_kappa(rater_a_labels, rater_b_labels, weights=weights)

    n = len(rater_a_labels)
    raw_agreement = sum(
        1 for a, b in zip(rater_a_labels, rater_b_labels) if a == b
    )
    disagreement_types = Counter(
        (a, b) for a, b in zip(rater_a_labels, rater_b_labels) if a != b
    )

    result["raw_agreement"] = round(raw_agreement / n, 4) if n else 0.0
    result["disagreement_types"] = dict(disagreement_types)
    return result


# ---------------------------------------------------------------------------
# 8. Power analysis for sample size planning (section 2.7)
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


def hallucination_rate_power_analysis(
    baseline_rate: float,
    minimum_detectable_shift: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Compute the sample size required to detect a shift in hallucination
    rate with the specified statistical power.

    Listing 2.8 companion function — wraps compute_sample_size() using
    the parameter name used in the chapter text (minimum_detectable_shift
    instead of minimum_detectable_effect). Uses the two-proportion z-test
    formula for sample size estimation; the test is two-tailed (we care
    about both increases and decreases).

    Args:
        baseline_rate: Expected hallucination rate at baseline (0 to 1).
                       Example: 0.05 for a 5% baseline hallucination rate.
        minimum_detectable_shift: The smallest shift in rate you want to
                                  reliably detect. Example: 0.02 to detect
                                  a 2-percentage-point shift (from 5% to 7%).
        alpha: Type I error rate (significance level). Default 0.05.
        power: Desired statistical power (1 - Type II error rate).
               Default 0.80.

    Returns:
        dict with required sample size per group and interpretation.
        See compute_sample_size() for the full field list.
    """
    return compute_sample_size(
        baseline_rate=baseline_rate,
        minimum_detectable_effect=minimum_detectable_shift,
        alpha=alpha,
        power=power,
    )


# ---------------------------------------------------------------------------
# 9. CI/CD gate wrapper (section 2.8, Listing 2.9)
# ---------------------------------------------------------------------------

class HallucinationGate:
    """
    Gate-compatible hallucination scorer wrapper for CI/CD pipelines.

    Wraps a scorer (by default CombinedHallucinationScorer) exposing a
    `.score()` method that returns an object with a numeric
    hallucination-risk score, and turns it into a CI-gate-compatible
    batch check. Satisfies the three contracts described in section 2.8:

      1. Accepts a batch of (prompt, response) pairs and returns a
         structured result, not just a float.
      2. Supports a configurable threshold so teams can tighten or
         relax the bar without touching the scoring code.
      3. Exposes exit_code(), which is non-zero when the batch fail
         rate exceeds the threshold, so a pipeline runner can treat it
         as a gate failure.

    Parameters
    ----------
    scorer : object
        Any object exposing score(question=..., answer=..., context=...)
        -> an object with a `combined_score` or `score` attribute, where
        higher = more hallucination risk (e.g. CombinedHallucinationScorer
        or HallucinationMetric).
    threshold : float
        Maximum acceptable hallucination/risk score. A pair scoring above
        this threshold counts as a gate failure.
    """

    def __init__(self, scorer, threshold: float = 0.5) -> None:
        self.scorer = scorer
        self.threshold = threshold

    def _score_pair(self, prompt: str, response: str, context: list[str] | None = None):
        """Score one (prompt, response) pair, tolerating both
        CombinedHallucinationScorer- and HallucinationMetric-style
        `.score()` signatures."""
        try:
            return self.scorer.score(question=prompt, answer=response, context=context or [])
        except TypeError:
            return self.scorer.score(input_text=prompt, actual_output=response, context=context or [])

    def run(self, pairs: list[dict]) -> dict:
        """
        Run the gate over a batch of (prompt, response) pairs.

        Each pair dict must contain 'prompt' and 'response' keys, and may
        contain an optional 'context' key (list[str]).

        Returns
        -------
        dict with 'fail_rate', 'passed', and 'results' (the per-pair
        scoring objects).
        """
        results = [
            self._score_pair(p["prompt"], p["response"], p.get("context"))
            for p in pairs
        ]
        risk_scores = [
            getattr(r, "combined_score", getattr(r, "score", None)) for r in results
        ]
        fail_count = sum(1 for s in risk_scores if s is not None and s > self.threshold)
        fail_rate = round(fail_count / len(results), 4) if results else 0.0
        return {
            "fail_rate": fail_rate,
            "passed": fail_rate == 0.0,
            "results": results,
        }

    def exit_code(self, result: dict) -> int:
        """Return 0 (success) if the gate passed, 1 (failure) otherwise —
        the code a CI/CD pipeline step checks to decide whether to block
        the merge."""
        return 0 if result["passed"] else 1


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
    print(f"  Passed (<={combined.threshold})   : {combined.passed}")


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
        description="Chapter 2 companion — hallucination detection demos"
    )
    parser.add_argument(
        "--demo",
        choices=["faithfulness", "kappa", "power", "all"],
        default="all",
        help="Which demo to run. Default: all",
    )
    args = parser.parse_args()

    print("\nChapter 2: Detecting Hallucinations Before Your Users Do")
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
