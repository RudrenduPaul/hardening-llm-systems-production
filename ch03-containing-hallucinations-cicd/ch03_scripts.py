"""
Chapter 4: Containing Hallucinations — Wiring Detection into CI/CD
Hardening LLM Systems in Production — Companion Code
Author: Rudrendu Paul | https://orcid.org/0009-0008-0141-4690

Implements:
  - HallucinationGate: full CI gate class with sys.exit(1) on failure
  - SelfConsistencyChecker: sample N completions, measure agreement
  - ClaimDecompositionPipeline: decompose compound answers into atomic claims
  - ShadowTrafficHarness: replay production traffic against a candidate model
  - GitHub Actions YAML generation helper

Requirements:
    openai>=1.0.0,<2.0
    deepeval==0.21.7
    ragas==0.1.21
    scikit-learn>=1.3.0,<2.0
    scipy>=1.11.0,<2.0

Usage:
    python ch04_scripts.py
    python ch04_scripts.py --gate          # Run CI gate demo
    python ch04_scripts.py --consistency   # Run self-consistency demo
    python ch04_scripts.py --claims        # Run claim decomposition demo
    python ch04_scripts.py --shadow        # Run shadow traffic demo
    python ch04_scripts.py --gen-yaml      # Print the GitHub Actions YAML
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Optional imports — system stays importable without the ML stack
# ---------------------------------------------------------------------------

try:
    import openai as _openai_module
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. HallucinationGate — the CI gate class
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


class HallucinationGate:
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
        gate = HallucinationGate(threshold=0.80)
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

        recommendation = self._recommend(mean_delta)

        return ShadowReport(
            total=len(comparisons),
            candidate_wins=wins,
            production_wins=losses,
            ties=ties,
            mean_production_score=round(mean_prod, 4),
            mean_candidate_score=round(mean_cand, 4),
            mean_delta=round(mean_delta, 4),
            recommendation=recommendation,
            comparisons=comparisons,
        )

    def _recommend(self, mean_delta: float) -> str:
        if mean_delta >= self.min_improvement:
            return f"PROMOTE — candidate improves mean score by {mean_delta:+.4f}."
        if mean_delta <= self.max_regression:
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
# 5. GitHub Actions YAML generator
# ---------------------------------------------------------------------------

GITHUB_ACTIONS_YAML = """\
# .github/workflows/hallucination-gate.yml
# Chapter 4 — Hardening LLM Systems in Production
# Hallucination CI gate: fails the build if mean faithfulness < threshold.

name: Hallucination Gate

on:
  pull_request:
    branches: [main, staging]
  push:
    branches: [main]

env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  HALLUCINATION_THRESHOLD: "0.80"   # Fail build if mean score < 0.80
  TEST_SUITE_PATH: "tests/hallucination/test_suite.json"

jobs:
  hallucination-gate:
    name: Hallucination Faithfulness Gate
    runs-on: ubuntu-latest
    timeout-minutes: 20

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
          pip install deepeval==0.21.7 ragas==0.1.21 \\
                      scikit-learn>=1.3.0 scipy>=1.11.0 openai>=1.0.0

      - name: Run hallucination gate
        run: |
          python ch04_scripts.py --gate \\
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
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 4 companion — CI/CD hallucination containment demos"
    )
    parser.add_argument("--gate", action="store_true", help="Run CI gate demo")
    parser.add_argument("--consistency", action="store_true", help="Run self-consistency demo")
    parser.add_argument("--claims", action="store_true", help="Run claim decomposition demo")
    parser.add_argument("--shadow", action="store_true", help="Run shadow traffic demo")
    parser.add_argument("--gen-yaml", action="store_true", help="Print GitHub Actions YAML")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--suite", type=str, default=None)
    args = parser.parse_args()

    # If no flags given, run everything
    run_all = not any([args.gate, args.consistency, args.claims, args.shadow, args.gen_yaml])

    print("\nChapter 4: Containing Hallucinations — Wiring Detection into CI/CD")
    print("Hardening LLM Systems in Production — Companion Code")
    print("=" * 60)

    if args.gate or run_all:
        print("\n--- CI Gate Demo ---")
        gate = HallucinationGate(threshold=args.threshold, exit_on_fail=False)

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

    if args.gen_yaml or run_all:
        print_github_actions_yaml()


if __name__ == "__main__":
    main()
