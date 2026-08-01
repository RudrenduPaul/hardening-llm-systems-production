"""
Chapter 10: Harmful Output and Bias: Detection, Thresholds, and the Deployment Gate
=====================================================================================
Stub implementations for the harmful-output and bias detection classes introduced
in Chapter 10. Full logic is in ch10_notebook.ipynb; this file is the importable
module form and the entry point for the CI/CD gate.

requires: nemoguardrails==0.9.1, guardrails-ai==0.4.5, openai>=1.30.0,<2.0, scikit-learn>=1.5.0,<2.0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

@dataclass
class HarmfulOutputTaxonomy:
    """Maps output categories to severity scores and regulatory liability.

    Each entry cross-references:
    - OWASP Top 10 for LLMs (LLM09 Misinformation, LLM06 Sensitive Information)
    - GDPR Article 22 (automated decision-making)
    - EU AI Act Annex III (high-risk AI system categories)

    Stub — full implementation in notebook.
    """

    category: str
    severity: float  # 0.0 (low) to 1.0 (critical)
    owasp_ref: str
    regulatory_refs: list[str] = field(default_factory=list)

    def is_blocking(self, threshold: float = 0.7) -> bool:
        """Return True when severity is at or above the blocking threshold."""
        raise NotImplementedError("See ch10_notebook.ipynb for the full implementation.")


# ---------------------------------------------------------------------------
# Runtime classifiers
# ---------------------------------------------------------------------------

class NeMoContentClassifier:
    """Colang policy-based content classifier using NeMo Guardrails.

    Loads a Colang policy file and routes each output through the async
    NeMo Guardrails engine for hate-speech and harmful-content detection.

    Stub — full implementation in notebook.
    """

    def __init__(self, policy_path: str | None = None) -> None:
        self.policy_path = policy_path

    def classify(self, text: str) -> dict[str, Any]:
        """Return a classification dict with keys: category, score, blocked.

        Stub — full implementation in notebook.
        """
        raise NotImplementedError("See ch10_notebook.ipynb for the full implementation.")


class GuardrailsAIValidator:
    """ToxicLanguage + DetectPII validator chain using Guardrails AI.

    Implements reask logic: when a validator fires, the chain re-prompts the
    model with the violation highlighted before returning a final response.

    Stub — full implementation in notebook.
    """

    def validate(self, text: str) -> dict[str, Any]:
        """Return validation result with keys: passed, violations, reask_needed.

        Stub — full implementation in notebook.
        """
        raise NotImplementedError("See ch10_notebook.ipynb for the full implementation.")


# ---------------------------------------------------------------------------
# SLO enforcement
# ---------------------------------------------------------------------------

class HarmfulContentSLO:
    """Enforces P99 latency budget and fail-safe/fail-open dispatch policy.

    When the classifier exceeds its latency budget, the SLO dispatches either:
    - fail-safe: block the output (conservative)
    - fail-open: allow the output (permissive)

    The policy is configurable per deployment environment.

    Stub — full implementation in notebook.
    """

    def __init__(self, p99_budget_ms: float = 200.0, policy: str = "fail-safe") -> None:
        self.p99_budget_ms = p99_budget_ms
        self.policy = policy

    def check(self, classifier_result: dict[str, Any], latency_ms: float) -> bool:
        """Return True when the output is safe to serve under the SLO policy.

        Stub — full implementation in notebook.
        """
        raise NotImplementedError("See ch10_notebook.ipynb for the full implementation.")


# ---------------------------------------------------------------------------
# Bias measurement
# ---------------------------------------------------------------------------

class CounterfactualBiasProbe:
    """Measures representational bias via counterfactual sentence pairs.

    Generates matched sentence pairs that differ only in a demographic
    attribute (gender, race, age) and scores the gap in model outputs.
    Also runs occupational association tests for systematic occupation-gender
    associations.

    Stub — full implementation in notebook.
    """

    def probe(self, model_fn: Any, attributes: list[str]) -> dict[str, float]:
        """Return a dict mapping attribute names to bias gap scores.

        Stub — full implementation in notebook.
        """
        raise NotImplementedError("See ch10_notebook.ipynb for the full implementation.")


class BiasJudgeCalibrator:
    """Calibrates an LLM-as-judge scorer using Cohen's kappa.

    Compares judge labels to a human-annotated reference set and reports
    kappa + a sklearn classification report. Recalibration raises or lowers
    the judge's internal thresholds until kappa meets the target.

    Stub — full implementation in notebook.
    """

    def calibrate(
        self,
        judge_labels: list[int],
        reference_labels: list[int],
        target_kappa: float = 0.7,
    ) -> dict[str, Any]:
        """Return calibration results including kappa score and report.

        Stub — full implementation in notebook.
        """
        raise NotImplementedError("See ch10_notebook.ipynb for the full implementation.")


# ---------------------------------------------------------------------------
# CI/CD gate
# ---------------------------------------------------------------------------

class HarmfulContentCIGate:
    """Unified CI/CD gate blocking on harmful fraction, calibration drift, or bias gap.

    Exit codes:
    - 0: all checks pass
    - 1: harmful_fraction > threshold OR bias_gap > 0.15 OR kappa < kappa_floor

    Stub — full implementation in notebook.
    """

    def __init__(
        self,
        harmful_fraction_threshold: float = 0.01,
        bias_gap_threshold: float = 0.15,
        kappa_floor: float = 0.7,
    ) -> None:
        self.harmful_fraction_threshold = harmful_fraction_threshold
        self.bias_gap_threshold = bias_gap_threshold
        self.kappa_floor = kappa_floor

    def run(
        self,
        classifier: NeMoContentClassifier | GuardrailsAIValidator,
        bias_probe: CounterfactualBiasProbe,
        calibrator: BiasJudgeCalibrator,
        samples: list[str],
    ) -> dict[str, Any]:
        """Execute all checks and return a structured gate report.

        Stub — full implementation in notebook.
        """
        raise NotImplementedError("See ch10_notebook.ipynb for the full implementation.")

    def exit_code(self, report: dict[str, Any]) -> int:
        """Return 0 if gate passes, 1 if any check fails."""
        raise NotImplementedError("See ch10_notebook.ipynb for the full implementation.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ch10_scripts",
        description="Chapter 10 harmful-output and bias CI gate — stubs only, see notebook.",
    )
    parser.add_argument(
        "--mode",
        choices=["gate", "report"],
        default="report",
        help="gate: run CI gate and exit 1 on failure. report: print report only.",
    )
    parser.add_argument(
        "--harmful-threshold",
        type=float,
        default=0.01,
        help="Maximum allowed harmful output fraction (default: 0.01).",
    )
    parser.add_argument(
        "--bias-gap",
        type=float,
        default=0.15,
        help="Maximum allowed bias gap score (default: 0.15).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    print(
        "ch10_scripts stubs loaded. "
        "Run ch10_notebook.ipynb for the full implementation."
    )
    if args.mode == "gate":
        print("--mode gate: full implementation required. Exiting 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
