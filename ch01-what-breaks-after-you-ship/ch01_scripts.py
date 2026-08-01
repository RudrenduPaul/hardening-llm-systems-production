"""
Chapter 1: What Breaks After You Ship
Hardening LLM Systems in Production — Companion Code
Author: Rudrendu Paul | https://orcid.org/0009-0008-0141-4690

This module implements the HardeningReport self-diagnostic scorecard: 25 yes/no questions
across 5 hardening vectors (hallucination containment, adversarial hardening, agentic
safety, data leakage prevention, compliance readiness — section 1.7). Run it against any
LLM-powered system to get a 0-25 exposure score and a prioritized remediation plan before
incidents occur in production.

Requirements:
    python>=3.11
    dataclasses (stdlib)
    json (stdlib)
    argparse (stdlib)

Usage:
    python ch01_scripts.py
    python ch01_scripts.py --interactive
    python ch01_scripts.py --export report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Exposure(str, Enum):
    """The three exposure tiers from section 1.7.3, keyed to the 0-25 total score."""
    CRITICAL = "Critical exposure"
    PARTIAL = "Partial hardening"
    DEFENSIBLE = "Defensible posture"


VECTOR_LABELS: dict[str, str] = {
    "hallucination_containment": "Hallucination containment",
    "adversarial_hardening": "Adversarial hardening",
    "agentic_safety": "Agentic safety",
    "data_leakage_prevention": "Data leakage prevention",
    "compliance_readiness": "Compliance readiness",
}

# Each vector's primary chapter(s), per section 1.7.1's vector-to-case-study mapping.
VECTOR_CHAPTER_REFERENCE: dict[str, str] = {
    "hallucination_containment": "Chapters 2-3",
    "adversarial_hardening": "Chapters 4, 6",
    "agentic_safety": "Chapter 7",
    "data_leakage_prevention": "Chapters 5, 8",
    "compliance_readiness": "Chapter 10",
}

# The 25 scorecard statements, five per vector, matching Listing 1.5 (section 1.7.2)
# in the manuscript exactly.
VECTOR_QUESTIONS: dict[str, list[str]] = {
    "hallucination_containment": [
        "Automated factuality evaluation (deepeval, RAGAS, or equivalent) runs against a maintained golden dataset before each release.",
        "RAG grounding is used for factual claims, with pipeline verification that the claim appears in retrieved context.",
        "Hallucination rate is defined as a deployment-blocking metric with a documented threshold.",
        "Live production outputs are sampled and run through factuality evaluation on a continuous basis.",
        "A documented rollback plan exists for a hallucination-rate spike in production.",
    ],
    "adversarial_hardening": [
        "An automated red-team scan (Garak, PyRIT, Promptfoo, or equivalent) ran against the current production endpoint in the past 90 days.",
        "Privilege separation is implemented: the LLM cannot access more resources than the current request requires.",
        "Output filtering catches exfiltration attempt patterns before responses reach the network.",
        "Injection detection tooling (Microsoft Prompt Shields, LLM Guard, or equivalent) runs in the request path.",
        "A documented manual red-team playbook covers failure classes that automation misses.",
    ],
    "agentic_safety": [
        "Scope enforcement for agents is implemented at the execution layer (tool-calling infrastructure), not only in the system prompt.",
        "Agents run with minimal privilege: access only to the resources the current task requires.",
        "Agents use stateless tool execution, without persistent memory that carries authority across sessions.",
        "Tripwires or anomaly detection on agent action sequences trigger human review above a defined risk threshold.",
        "A documented incident-response playbook exists specifically for agent-scope violations.",
    ],
    "data_leakage_prevention": [
        "PII detection (Microsoft Presidio or equivalent) runs on inputs before they enter the LLM context.",
        "PII detection runs on LLM outputs before they are displayed or stored.",
        "The vector store or retrieval database is behind authentication that enforces tenant isolation.",
        "A documented right-to-erasure implementation exists for conversation logs under GDPR and CCPA.",
        "A data-tier security review (network reachability, authentication, access controls) has run on all databases that store LLM interaction logs.",
    ],
    "compliance_readiness": [
        "The LLM application has been classified against the EU AI Act Annex III risk tiers.",
        "The system generates and maintains the Annex IV technical documentation package.",
        "A post-market monitoring pipeline exists with a process for detecting serious incidents within the 15-day reporting window (EU AI Act Article 73).",
        "System controls have been mapped to the NIST AI RMF and a profile document produced.",
        "The CI/CD pipeline includes an Annex IV artifact completeness check as a merge-blocking gate.",
    ],
}

_VECTOR_PREFIX: dict[str, str] = {
    "hallucination_containment": "HC",
    "adversarial_hardening": "AH",
    "agentic_safety": "AS",
    "data_leakage_prevention": "DL",
    "compliance_readiness": "CR",
}

_RATIONALE: dict[str, str] = {
    "HC1": "Without an automated gate, hallucination regressions ship silently.",
    "HC2": "Ungrounded procedural claims are exactly the failure that produced the Moffatt v. Air Canada ruling.",
    "HC3": "A metric nobody blocks on is a dashboard, not a control.",
    "HC4": "Pre-launch evaluation catches nothing that appears only under real production traffic.",
    "HC5": "Detecting a spike without a rehearsed rollback path costs the response time that matters most.",
    "AH1": "Prompt injection techniques evolve continuously; a scan older than a quarter is stale evidence.",
    "AH2": "EchoLeak's blast radius was every document Copilot could read, not just the one email it processed.",
    "AH3": "Without an output filter, a successful injection has an unobstructed path out.",
    "AH4": "Detection at the request boundary catches attempts that privilege separation alone does not stop.",
    "AH5": "Automated scanners miss the novel attack patterns that a structured manual review catches.",
    "AS1": "System-prompt-only scope is advisory; the execution layer is the only enforcement with physical authority.",
    "AS2": "Excess privilege turns a single bad tool call into a system-wide incident.",
    "AS3": "Persistent memory that carries authority lets one session's compromise propagate into future sessions.",
    "AS4": "Without a tripwire, an agent's out-of-scope action sequence completes before anyone notices.",
    "AS5": "A generic incident playbook doesn't cover the specific containment steps an agent-scope violation needs.",
    "DL1": "PII that enters the context window can be reflected back in outputs or logged downstream.",
    "DL2": "Models can surface PII that never appeared in the current input, drawn from training data or retrieved context.",
    "DL3": "An unauthenticated retrieval store is the exact failure that produced the DeepSeek exposure.",
    "DL4": "A deletion request you cannot fulfill is a compliance failure, not just an engineering gap.",
    "DL5": "Interaction logs accumulate the same sensitive content the DeepSeek exposure leaked, and rarely get the same scrutiny as the primary application database.",
    "CR1": "Every other compliance obligation follows from the risk-tier classification.",
    "CR2": "The technical file is what a regulator requests first after an incident, not something to assemble after the fact.",
    "CR3": "The 15-day window starts at detection, not at documentation; a slow detection pipeline consumes the window before the compliance team sees the incident.",
    "CR4": "Without a mapping, you cannot show a reviewer which control addresses which framework requirement.",
    "CR5": "A documentation package that isn't enforced in CI/CD drifts out of date the first time a control changes.",
}

_REMEDIATION: dict[str, str] = {
    "HC1": "Add a faithfulness score threshold to your CI pipeline. Chapters 2 and 3 build the detection pipeline and wire it into CI/CD.",
    "HC2": "Force procedural answers to cite retrieved context; reject claims with no supporting passage.",
    "HC3": "Chapter 3 wires the hallucination-rate threshold into the CI/CD gate as a merge-blocking signal.",
    "HC4": "Sample a percentage of live outputs and route them through the same evaluation pipeline used in CI.",
    "HC5": "Version model and prompt configurations, and rehearse the rollback procedure before you need it.",
    "AH1": "Schedule a recurring red-team sprint using the framework Chapter 6 builds.",
    "AH2": "Scope every retrieval and tool call to the minimum resource set the current request needs. Chapter 4 builds the architecture.",
    "AH3": "Add an output filter that inspects responses for exfiltration patterns before they leave the application boundary.",
    "AH4": "Deploy an injection classifier in front of the model call and log every flagged request.",
    "AH5": "Maintain a manual red-team playbook alongside automated scans; Chapter 6 provides the structure.",
    "AS1": "Move scope enforcement to the tool-calling infrastructure. Chapter 7 builds the execution-layer containment.",
    "AS2": "Define per-task tool allow-lists and audit grants on a recurring schedule.",
    "AS3": "Reset agent authority at the start of every session; never let memory grant privilege.",
    "AS4": "Instrument agent action sequences with anomaly scoring and route high-risk sequences to human review.",
    "AS5": "Write a dedicated agent-scope-violation runbook; Chapter 7 provides a template.",
    "DL1": "Deploy a PII detector ahead of the model call. Chapter 8 covers detection and redaction.",
    "DL2": "Mirror the input-side PII detector on the output path before responses reach the user or a log.",
    "DL3": "Require authentication on every retrieval store and enforce tenant boundaries at the query layer. Chapter 5 builds this.",
    "DL4": "Implement and test a right-to-erasure pipeline against conversation logs. Chapter 8 covers the implementation.",
    "DL5": "Run a network-reachability and access-control review against every database storing LLM interaction data.",
    "CR1": "Classify the application against Annex III now; Chapter 10 walks the classification process.",
    "CR2": "Automate Annex IV artifact generation from existing CI/CD outputs. Chapter 10 builds the pipeline.",
    "CR3": "Build monitoring that flags a serious incident automatically and starts the 15-day clock at detection.",
    "CR4": "Produce a NIST AI RMF profile document mapping each engineering control to its framework function.",
    "CR5": "Add an Annex IV completeness check to the merge pipeline. Chapter 10 wires this as the final gate.",
}


def _build_diagnostic_questions() -> list[dict]:
    questions: list[dict] = []
    for vector_key, statements in VECTOR_QUESTIONS.items():
        prefix = _VECTOR_PREFIX[vector_key]
        for i, statement in enumerate(statements, start=1):
            qid = f"{prefix}{i}"
            questions.append({
                "id": qid,
                "vector": vector_key,
                "question": statement,
                "rationale": _RATIONALE[qid],
                "remediation": _REMEDIATION[qid],
            })
    return questions


# 25 questions: 5 per vector, in the same order as VECTOR_QUESTIONS.
DIAGNOSTIC_QUESTIONS: list[dict] = _build_diagnostic_questions()


# ---------------------------------------------------------------------------
# Core scorecard class
# ---------------------------------------------------------------------------

@dataclass
class VectorScore:
    key: str
    label: str
    yes_count: int = 0
    total: int = 0
    gap_questions: list[dict] = field(default_factory=list)

    @property
    def score_pct(self) -> int:
        return round(self.yes_count / self.total * 100) if self.total else 0


@dataclass
class HardeningReport:
    """
    Self-diagnostic scorecard for LLM systems in production (Listing 1.5, section 1.7).

    Covers 25 questions across 5 hardening vectors. Each question is answered
    yes (1) or no (0). The total (0-25) drives a 3-tier Exposure classification
    (Critical exposure / Partial hardening / Defensible posture, section 1.7.3)
    and a prioritized remediation plan targeting the most dangerous gaps first.

    Usage
    -----
    >>> report = HardeningReport()
    >>> report.answer("HC1", True)
    >>> report.answer("HC2", False)
    >>> print(report.summary())

    Or run the full interactive diagnostic:
    >>> report = HardeningReport.run_interactive()

    Or build one programmatically from a pre-filled answer dict:
    >>> report = HardeningReport.score_from_dict({"HC1": True, "HC2": False})
    """

    answers: dict[str, bool] = field(default_factory=dict)
    system_name: str = "My LLM System"
    assessor: str = "Unknown"

    # ------------------------------------------------------------------
    # Answering questions
    # ------------------------------------------------------------------

    def answer(self, question_id: str, response: bool) -> None:
        """Record a yes/no answer for a question ID (e.g. 'HC1', 'AH3')."""
        valid_ids = {q["id"] for q in DIAGNOSTIC_QUESTIONS}
        if question_id not in valid_ids:
            raise ValueError(f"Unknown question ID '{question_id}'. Valid IDs: {sorted(valid_ids)}")
        self.answers[question_id] = response

    def answer_all_no(self) -> None:
        """Seed all answers as 'no' — useful for demonstrating worst-case."""
        for q in DIAGNOSTIC_QUESTIONS:
            self.answers[q["id"]] = False

    def answer_all_yes(self) -> None:
        """Seed all answers as 'yes' — useful for demonstrating best-case."""
        for q in DIAGNOSTIC_QUESTIONS:
            self.answers[q["id"]] = True

    @classmethod
    def score_from_dict(
        cls,
        answers: dict[str, bool],
        system_name: str = "My LLM System",
        assessor: str = "Unknown",
    ) -> "HardeningReport":
        """
        Build a HardeningReport from a pre-filled answer dict.

        The programmatic counterpart to run_interactive(): pass a dict mapping
        question IDs (e.g. 'HC1', 'AH3') to bool answers and get back a scored
        report with no terminal interaction. Useful for CI jobs and
        new-feature onboarding checklists where the answers come from a
        config file, a form submission, or another automated source rather
        than an interactive prompt.

        Parameters
        ----------
        answers:
            dict mapping question IDs to bool responses. Every key is
            validated against DIAGNOSTIC_QUESTIONS via `answer()`.
        system_name, assessor:
            Same metadata fields as the HardeningReport constructor.

        Raises
        ------
        ValueError if any key in `answers` is not a valid question ID.
        """
        report = cls(system_name=system_name, assessor=assessor)
        for question_id, response in answers.items():
            report.answer(question_id, response)
        return report

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_vector_scores(self) -> dict[str, VectorScore]:
        scores: dict[str, VectorScore] = {
            key: VectorScore(key=key, label=VECTOR_LABELS[key]) for key in VECTOR_QUESTIONS
        }
        for q in DIAGNOSTIC_QUESTIONS:
            vector_score = scores[q["vector"]]
            vector_score.total += 1
            answered = self.answers.get(q["id"])
            if answered is True:
                vector_score.yes_count += 1
            elif answered is False:
                vector_score.gap_questions.append(q)
        return scores

    @property
    def max_score(self) -> int:
        return len(DIAGNOSTIC_QUESTIONS)

    @property
    def total_score(self) -> int:
        """Total points: 1 per 'yes' answer, out of 25 (section 1.7.3)."""
        return sum(1 for v in self.answers.values() if v)

    @property
    def total_score_pct(self) -> int:
        return round(self.total_score / self.max_score * 100) if self.max_score else 0

    @property
    def exposure_tier(self) -> Exposure:
        """
        3-tier exposure classification from the 0-25 total score (section 1.7.3):
        0-10 Critical exposure, 11-18 Partial hardening, 19-25 Defensible posture.
        """
        if self.total_score >= 19:
            return Exposure.DEFENSIBLE
        if self.total_score >= 11:
            return Exposure.PARTIAL
        return Exposure.CRITICAL

    @property
    def answered_count(self) -> int:
        return len(self.answers)

    def vector_results(self) -> list[dict]:
        """
        Per-vector score, gap, and chapter-reference summary, in vector order.

        This is the single source of truth for per-vector results —
        HardeningReadinessReport.from_report() (Listing 1.6) consumes this
        directly rather than re-deriving a second, parallel classification
        from the raw answers dict (see section 1.7.5).
        """
        scores = self._compute_vector_scores()
        return [
            {
                "key": key,
                "label": vs.label,
                "score": vs.yes_count,
                "max_score": vs.total,
                "chapter_reference": VECTOR_CHAPTER_REFERENCE[key],
                "gaps": [q["id"] for q in vs.gap_questions],
            }
            for key, vs in scores.items()
        ]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted text summary of the diagnostic results."""
        vector_scores = self._compute_vector_scores()
        lines: list[str] = []

        lines.append("=" * 68)
        lines.append("  HARDENING REPORT: LLM System Self-Diagnostic")
        lines.append("=" * 68)
        lines.append(f"  System  : {self.system_name}")
        lines.append(f"  Assessor: {self.assessor}")
        lines.append(f"  Date    : {_today()}")
        lines.append("")
        lines.append(f"  OVERALL SCORE : {self.total_score}/{self.max_score} ({self.total_score_pct}%, {self.answered_count}/25 questions answered)")
        lines.append(f"  EXPOSURE TIER : {self.exposure_tier.value}")
        lines.append("")
        lines.append("-" * 68)
        lines.append("  VECTOR SCORES")
        lines.append("-" * 68)

        for vs in vector_scores.values():
            bar = _progress_bar(vs.yes_count / vs.total if vs.total else 0.0, width=20)
            lines.append(f"  {vs.label:<28} {bar} {vs.yes_count}/{vs.total}  ({vs.score_pct}%)")

        lines.append("")
        lines.append("-" * 68)
        lines.append("  PRIORITIZED REMEDIATION PLAN")
        lines.append("  (ordered by vector score, lowest first)")
        lines.append("-" * 68)

        # Sort vectors by ascending score (worst first)
        sorted_vectors = sorted(vector_scores.values(), key=lambda vs: vs.yes_count)
        item_num = 1
        for vs in sorted_vectors:
            if not vs.gap_questions:
                continue
            lines.append(f"\n  {vs.label.upper()} — {vs.yes_count}/{vs.total} ({vs.score_pct}%)")
            for q in vs.gap_questions:
                lines.append(f"    [{item_num:02d}] {q['id']}: {q['question']}")
                lines.append(f"         Why it matters: {q['rationale']}")
                lines.append(f"         Fix: {q['remediation']}")
                item_num += 1

        if item_num == 1:
            lines.append("\n  No gaps found. All 25 controls are in place.")

        lines.append("")
        lines.append("=" * 68)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize the full report to a dict (JSON-serializable)."""
        return {
            "system_name": self.system_name,
            "assessor": self.assessor,
            "date": _today(),
            "total_score": self.total_score,
            "max_score": self.max_score,
            "overall_score_pct": self.total_score_pct,
            "exposure_tier": self.exposure_tier.name,
            "exposure_tier_label": self.exposure_tier.value,
            "answered_questions": self.answered_count,
            "total_questions": len(DIAGNOSTIC_QUESTIONS),
            "vectors": self.vector_results(),
            "answers": self.answers,
        }

    def export_json(self, path: str) -> None:
        """Write the report to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        print(f"Report saved to {path}")

    # ------------------------------------------------------------------
    # Interactive mode
    # ------------------------------------------------------------------

    @classmethod
    def run_interactive(cls, system_name: Optional[str] = None) -> "HardeningReport":
        """
        Run the full 25-question diagnostic interactively via stdin.

        Returns the completed HardeningReport for further use.
        """
        print("\n" + "=" * 68)
        print("  LLM HARDENING SELF-DIAGNOSTIC — Interactive Mode")
        print("=" * 68)

        if system_name is None:
            system_name = input("  System name (press Enter to skip): ").strip() or "My LLM System"

        assessor = input("  Your name (press Enter to skip): ").strip() or "Unknown"
        report = cls(system_name=system_name, assessor=assessor)

        print(f"\n  Answer each question with y/n (or yes/no).\n")

        current_vector = None
        for q in DIAGNOSTIC_QUESTIONS:
            if q["vector"] != current_vector:
                current_vector = q["vector"]
                print(f"\n  == {VECTOR_LABELS[current_vector].upper()} ==")

            while True:
                raw = input(f"  [{q['id']}] {q['question']}\n  > ").strip().lower()
                if raw in ("y", "yes"):
                    report.answer(q["id"], True)
                    break
                if raw in ("n", "no"):
                    report.answer(q["id"], False)
                    break
                print("  Please enter y or n.")

        return report


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    from datetime import date
    return date.today().isoformat()


def _progress_bar(ratio: float, width: int = 20) -> str:
    filled = round(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def demo_report(scenario: str = "mixed") -> HardeningReport:
    """
    Build a pre-filled demo report.

    Parameters
    ----------
    scenario : "mixed" | "best" | "worst"
        "mixed" — realistic mid-stage team, lands in "Partial hardening" (default)
        "best"  — all controls in place ("Defensible posture")
        "worst" — no controls in place ("Critical exposure")
    """
    report = HardeningReport(
        system_name="Acme Customer Support Bot v2.1",
        assessor="Rudrendu Paul",
    )

    if scenario == "best":
        report.answer_all_yes()
        return report

    if scenario == "worst":
        report.answer_all_no()
        return report

    # Mixed scenario: a realistic early-production LLM system. Total = 12/25,
    # which lands in the "Partial hardening" tier (11-18, section 1.7.3).
    mixed_answers: dict[str, bool] = {
        # Hallucination containment: CI check, RAG grounding, and threshold
        # exist; no continuous production sampling, no rollback plan.
        "HC1": True,  "HC2": True,  "HC3": True,  "HC4": False, "HC5": False,
        # Adversarial hardening: red-team scan and manual playbook exist;
        # no privilege separation, no output filtering, no injection tooling.
        "AH1": True,  "AH2": False, "AH3": False, "AH4": False, "AH5": True,
        # Agentic safety: minimal privilege and stateless execution in place;
        # no execution-layer scope enforcement, no tripwires, no runbook.
        "AS1": False, "AS2": True,  "AS3": True,  "AS4": False, "AS5": False,
        # Data leakage prevention: input PII detection and store auth exist;
        # no output-side PII scan, no right-to-erasure, no data-tier review.
        "DL1": True,  "DL2": False, "DL3": True,  "DL4": False, "DL5": False,
        # Compliance readiness: Annex III classification and NIST RMF mapping
        # exist; no Annex IV package, no monitoring pipeline, no CI/CD gate.
        "CR1": True,  "CR2": False, "CR3": False, "CR4": True,  "CR5": False,
    }
    for qid, ans in mixed_answers.items():
        report.answer(qid, ans)
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Hardening Self-Diagnostic — Chapter 1 companion script"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run the full 25-question diagnostic interactively.",
    )
    parser.add_argument(
        "--demo",
        choices=["mixed", "best", "worst"],
        default=None,
        help="Print a pre-filled demo report without interactive input.",
    )
    parser.add_argument(
        "--export",
        metavar="PATH",
        default=None,
        help="Export report to a JSON file at the given path.",
    )
    args = parser.parse_args()

    if args.interactive:
        report = HardeningReport.run_interactive()
    else:
        scenario = args.demo if args.demo else "mixed"
        print(f"\n[demo mode — scenario: {scenario}]")
        report = demo_report(scenario)

    print(report.summary())

    if args.export:
        report.export_json(args.export)


if __name__ == "__main__":
    main()


# === Listings 1.1-1.6: NVD scanner, incident tagger, property classifier, exposure map, readiness report ===

# ---------------------------------------------------------------------------
# Listing 1.1: scan_llm_stack — NIST NVD CVE scanner for LLM stack libraries
# Requirements: requests>=2.31.0,<3.0
# NIST NVD API v2 allows 5 requests per 30 seconds without an API key.
# Register at https://nvd.nist.gov/developers/request-an-api-key for higher limits.
# ---------------------------------------------------------------------------

import time as _time
from dataclasses import dataclass as _dataclass, field as _field
from datetime import datetime as _datetime
from typing import Optional as _Optional

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

DEFAULT_LLM_STACK: list[str] = [
    "langchain",
    "langchain-core",
    "langchain-community",
    "llama-index",
    "openai",
    "anthropic",
    "chromadb",
    "faiss-cpu",
    "tiktoken",
    "transformers",
    "sentence-transformers",
    "requests",
    "pydantic",
]


@_dataclass
class CVEFinding:
    """A single CVE finding for a library in the LLM stack."""
    library: str
    cve_id: str
    description: str
    cvss_score: _Optional[float]
    severity: str          # "CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"
    is_kev: bool           # True if in CISA Known Exploited Vulnerabilities catalog
    published: str


@_dataclass
class StackScanReport:
    """Aggregated result of scanning all libraries in the LLM stack."""
    libraries_scanned: int
    total_cves: int
    kev_count: int
    critical_count: int
    high_count: int
    findings: list[CVEFinding] = _field(default_factory=list)
    scan_timestamp: str = _field(default_factory=lambda: _datetime.utcnow().isoformat())


def _fetch_kev_set(api_key: _Optional[str] = None) -> set[str]:
    """Fetch the CISA KEV catalog and return a set of CVE IDs."""
    if not _REQUESTS_AVAILABLE:
        return set()
    try:
        headers = {"apiKey": api_key} if api_key else {}
        resp = _requests.get(_CISA_KEV_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return {v["cveID"] for v in data.get("vulnerabilities", [])}
    except Exception as exc:
        print(f"[NVD] KEV fetch failed: {exc}")
        return set()


def _fetch_cves_for_library(
    library: str,
    api_key: _Optional[str] = None,
) -> list[dict]:
    """Query NVD API v2 for CVEs matching a library keyword."""
    if not _REQUESTS_AVAILABLE:
        return []
    headers = {}
    if api_key:
        headers["apiKey"] = api_key
    params = {"keywordSearch": library, "resultsPerPage": 20}
    try:
        resp = _requests.get(_NVD_BASE, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("vulnerabilities", [])
    except Exception as exc:
        print(f"[NVD] Query for '{library}' failed: {exc}")
        return []


def scan_llm_stack(
    libraries: list[str] = None,
    api_key: _Optional[str] = None,
    requests_per_window: int = 5,
    window_seconds: int = 30,
) -> StackScanReport:
    """
    Scan a list of LLM stack library names against the NIST NVD for known CVEs.

    Cross-references results against the CISA Known Exploited Vulnerabilities (KEV)
    catalog.  A KEV-flagged CVE warrants immediate remediation regardless of CVSS score.

    Parameters
    ----------
    libraries:
        Library names to scan.  Defaults to DEFAULT_LLM_STACK.
    api_key:
        Optional NIST NVD API key.  Without one, the rate limit is 5 requests / 30s.
        Register at https://nvd.nist.gov/developers/request-an-api-key.
    requests_per_window:
        Number of requests allowed per window.  Default 5 (unauthenticated).
        Set to 50 if you have an API key.
    window_seconds:
        Rate-limit window in seconds.  Default 30.

    Returns
    -------
    StackScanReport with all findings, sorted by severity (CRITICAL first).
    """
    if libraries is None:
        libraries = DEFAULT_LLM_STACK

    print(f"[NVD] Fetching CISA KEV catalog …")
    kev_set = _fetch_kev_set(api_key)
    print(f"[NVD] KEV catalog loaded: {len(kev_set)} entries.")

    all_findings: list[CVEFinding] = []
    batch_count = 0

    for idx, library in enumerate(libraries):
        if batch_count >= requests_per_window:
            print(f"[NVD] Rate-limit pause: {window_seconds}s …")
            _time.sleep(window_seconds)
            batch_count = 0

        print(f"[NVD] Scanning '{library}' ({idx + 1}/{len(libraries)}) …")
        raw_cves = _fetch_cves_for_library(library, api_key)
        batch_count += 1

        for entry in raw_cves:
            cve = entry.get("cve", {})
            cve_id = cve.get("id", "")
            descriptions = cve.get("descriptions", [])
            desc = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
            published = cve.get("published", "")[:10]

            metrics = cve.get("metrics", {})
            cvss_score = None
            severity = "NONE"
            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                if key in metrics and metrics[key]:
                    cvss_data = metrics[key][0].get("cvssData", {})
                    cvss_score = cvss_data.get("baseScore")
                    severity = cvss_data.get("baseSeverity", "NONE")
                    break

            is_kev = cve_id in kev_set
            all_findings.append(CVEFinding(
                library=library,
                cve_id=cve_id,
                description=desc[:200],
                cvss_score=cvss_score,
                severity=severity.upper(),
                is_kev=is_kev,
                published=published,
            ))

    # Sort: KEV first, then by CVSS descending
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4}
    all_findings.sort(
        key=lambda f: (not f.is_kev, severity_order.get(f.severity, 5), -(f.cvss_score or 0))
    )

    return StackScanReport(
        libraries_scanned=len(libraries),
        total_cves=len(all_findings),
        kev_count=sum(1 for f in all_findings if f.is_kev),
        critical_count=sum(1 for f in all_findings if f.severity == "CRITICAL"),
        high_count=sum(1 for f in all_findings if f.severity == "HIGH"),
        findings=all_findings,
    )


# ---------------------------------------------------------------------------
# Listing 1.2: tag_incident — TF-IDF archetype tagger for production incidents
# Requirements: scikit-learn>=1.4.0,<2.0
# ---------------------------------------------------------------------------

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# Reference corpus for the four chapter archetypes.
# Distilled from sections 1.1–1.4. Extend with your own post-mortems.
_ARCHETYPE_CORPUS: dict[str, str] = {
    "hallucination-liability": (
        "chatbot hallucination confabulation factual error wrong policy procedure "
        "user acted on incorrect information legal liability tribunal ruling "
        "bereavement fare discount wrong claim output validation factuality "
        "customer service incorrect response grounding retrieval augmented "
        "model generated plausible but false statement regulatory consequence"
    ),
    "indirect-injection": (
        "prompt injection indirect injection email content malicious instruction "
        "retrieval context trust boundary exfiltration data leak corporate "
        "copilot assistant tool call external content untrusted source "
        "CVE vulnerability attack surface instruction following blast radius "
        "privilege separation output filter network exfiltration RAG pipeline"
    ),
    "agent-scope-violation": (
        "agent autonomous action exceeded authorization scope system prompt override "
        "vendor master write unauthorized database modification tool calling "
        "execution layer trust external document invoice malformed instruction "
        "agentic misbehavior out-of-scope action write access stateless "
        "minimal privilege tripwire anomaly detection human review gate"
    ),
    "data-tier-exposure": (
        "database misconfiguration unauthenticated ClickHouse exposed API key "
        "conversation log chat history storage cloud misconfiguration public internet "
        "system prompt exposed backend operational data Wiz Research DeepSeek "
        "data breach vector store authentication network access control data tier "
        "sensitive user data credential exposure audit compliance data retention"
    ),
}


@_dataclass
class IncidentTagResult:
    """Result of tagging an incident report against the four archetypes."""
    primary_archetype: str
    similarity_scores: dict[str, float]
    confidence: str      # "high" (>0.4 gap), "medium" (0.2-0.4 gap), "low" (<0.2 gap)
    remediation_chapter: str


def tag_incident(incident_text: str) -> IncidentTagResult:
    """
    Tag a production incident report against the four GenAI failure archetypes.

    Uses TF-IDF vectorization and cosine similarity.  The classifier is
    deliberately lightweight: it uses keyword matching, not an LLM, so
    it produces the same result every time on the same input and has no
    external dependencies beyond scikit-learn.

    Parameters
    ----------
    incident_text:
        Free-text incident description, Slack message, or post-mortem excerpt.

    Returns
    -------
    IncidentTagResult with the best-matching archetype, full score vector,
    and the primary chapter that addresses it.
    """
    if not _SKLEARN_AVAILABLE:
        return IncidentTagResult(
            primary_archetype="sklearn-not-installed",
            similarity_scores={},
            confidence="low",
            remediation_chapter="Install scikit-learn>=1.4.0 to enable tagging.",
        )

    corpus_labels = list(_ARCHETYPE_CORPUS.keys())
    corpus_texts = list(_ARCHETYPE_CORPUS.values()) + [incident_text]

    vectorizer = _TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)

    incident_vec = tfidf_matrix[-1]
    archetype_vecs = tfidf_matrix[:-1]
    sims = _cosine_similarity(incident_vec, archetype_vecs).flatten()

    scores = {label: round(float(sim), 4) for label, sim in zip(corpus_labels, sims)}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = best_score - second_score

    confidence = "high" if gap >= 0.4 else ("medium" if gap >= 0.2 else "low")

    chapter_map = {
        "hallucination-liability": "Chapters 2–3 (hallucination detection and CI/CD gating)",
        "indirect-injection": "Chapters 4–5 (injection defense and RAG authorization)",
        "agent-scope-violation": "Chapter 7 (agent scope containment and telemetry)",
        "data-tier-exposure": "Chapters 5 and 8 (retrieval authorization and PII pipeline)",
    }

    return IncidentTagResult(
        primary_archetype=best_label,
        similarity_scores=scores,
        confidence=confidence,
        remediation_chapter=chapter_map.get(best_label, "See Appendix A"),
    )


# ---------------------------------------------------------------------------
# Listing 1.3: classify_incident — keyword-based GenAI risk property classifier
# Requirements: none (standard library only)
# ---------------------------------------------------------------------------

import re as _re

_PROPERTY_KEYWORDS: dict[str, list[str]] = {
    "non_determinism": [
        "inconsistent", "different output", "sometimes works", "intermittent",
        "flaky", "random", "stochastic", "temperature", "sampling", "varies",
        "not reproducible", "depends on run", "different result", "model version",
        "fine-tuning", "hardware", "floating point", "non-deterministic",
        "same input different", "regression broke", "output changed",
    ],
    "open_ended_output": [
        "unexpected output", "never anticipated", "said something it shouldn't",
        "hallucinated", "confabulated", "made up", "fabricated", "invented",
        "wrong claim", "incorrect statement", "false information", "wrong policy",
        "wrong procedure", "wrong amount", "wrong date", "wrong name",
        "outside expected range", "unanticipated", "not covered in tests",
        "output not in spec", "said anything", "free-form",
    ],
    "emergent_behavior": [
        "after upgrade", "model update", "new capability", "wasn't able to before",
        "now it can", "version change", "fine-tune introduced", "capability regression",
        "compliance review outdated", "wasn't tested at this scale",
        "behavior changed", "started doing something new", "emerged",
        "appeared after", "not in original review", "capability appeared",
    ],
    "instruction_following": [
        "prompt injection", "ignored instructions", "system prompt override",
        "jailbreak", "followed malicious instruction", "instruction in document",
        "external content executed", "bypassed safety", "role confusion",
        "persona injection", "instruction override", "embedded command",
        "indirect injection", "followed user instruction instead of system",
        "instruction from email", "content treated as command",
    ],
}


@_dataclass
class PropertyClassification:
    """Result of classifying an incident by its primary GenAI risk property."""
    primary_property: str
    score_vector: dict[str, int]
    confidence: str          # "high", "medium", "low"
    chapter_focus: str


def classify_incident(incident_text: str) -> PropertyClassification:
    """
    Classify a production incident by the GenAI risk property that drove it.

    Uses keyword counting rather than an LLM call so the classifier is fast,
    auditable, and deterministic.

    Parameters
    ----------
    incident_text:
        Free-text description of the incident.

    Returns
    -------
    PropertyClassification with the primary property, score vector, and the
    chapter most relevant to that property.
    """
    text_lower = incident_text.lower()
    scores: dict[str, int] = {}

    for prop, keywords in _PROPERTY_KEYWORDS.items():
        count = 0
        for kw in keywords:
            # Escape the keyword for regex word-boundary matching
            pattern = _re.escape(kw)
            count += len(_re.findall(pattern, text_lower))
        scores[prop] = count

    total = sum(scores.values())
    if total == 0:
        return PropertyClassification(
            primary_property="unknown",
            score_vector=scores,
            confidence="low",
            chapter_focus="Review sections 1.6 and 1.5 to identify the failure pattern manually.",
        )

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_prop, best_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0

    ratio = best_count / total
    confidence = "high" if ratio >= 0.6 else ("medium" if ratio >= 0.4 else "low")

    chapter_map = {
        "non_determinism": "Chapters 2–3 (statistical evaluation and shadow traffic)",
        "open_ended_output": "Chapters 2–3 (hallucination detection pipeline)",
        "emergent_behavior": "Chapter 1 section 1.6.3 + Chapter 10 (capability monitoring)",
        "instruction_following": "Chapters 4–6 (injection defense, red-teaming)",
    }

    return PropertyClassification(
        primary_property=best_prop,
        score_vector=scores,
        confidence=confidence,
        chapter_focus=chapter_map.get(best_prop, "See Appendix A"),
    )


# ---------------------------------------------------------------------------
# Listing 1.4: generate_exposure_map — post-deployment exposure map generator
# Requirements: dataclasses (stdlib), typing (stdlib)
# ---------------------------------------------------------------------------

from typing import Literal as _Literal

_DeploymentType = _Literal["chat", "rag", "agent", "api"]
_CoverageStatus = _Literal["covered", "gap", "single-point-of-failure", "governance-gap"]

_VECTOR_DESCRIPTIONS: dict[str, str] = {
    "hallucination_containment": "Factuality evaluation, golden dataset, CI gate",
    "adversarial_hardening": "Red-teaming, privilege separation, output filtering",
    "agentic_safety": "Execution-layer scope, minimal privilege, tripwires",
    "data_leakage_prevention": "PII detection, vector store auth, right-to-erasure",
    "compliance_readiness": "Annex IV artifacts, NIST AI RMF mapping, audit trail",
}


@_dataclass
class TeamRoles:
    """Names or role labels for each engineering function."""
    ml_engineer: _Optional[str] = None
    security_engineer: _Optional[str] = None
    data_engineer: _Optional[str] = None
    platform_engineer: _Optional[str] = None
    compliance_lead: _Optional[str] = None

    def filled_roles(self) -> list[str]:
        return [
            role for role in [
                self.ml_engineer,
                self.security_engineer,
                self.data_engineer,
                self.platform_engineer,
                self.compliance_lead,
            ]
            if role is not None
        ]


@_dataclass
class ExposureVectorStatus:
    """Coverage status and ownership for a single hardening vector."""
    vector_key: str
    description: str
    status: _CoverageStatus
    owner: _Optional[str]
    gap_reason: str = ""


@_dataclass
class ExposureMap:
    """Full post-deployment exposure map for an LLM system."""
    deployment_type: str
    team_roles: "TeamRoles"
    vectors: list[ExposureVectorStatus] = _field(default_factory=list)

    def single_point_of_failure_count(self) -> int:
        return sum(1 for v in self.vectors if v.status == "single-point-of-failure")

    def gap_count(self) -> int:
        return sum(1 for v in self.vectors if v.status == "gap")

    def governance_gap_count(self) -> int:
        return sum(1 for v in self.vectors if v.status == "governance-gap")

    def summary(self) -> str:
        lines: list[str] = [
            "=" * 64,
            f"  EXPOSURE MAP — Deployment type: {self.deployment_type}",
            "=" * 64,
        ]
        for vs in self.vectors:
            status_tag = {
                "covered": "[OK  ]",
                "gap": "[GAP ]",
                "single-point-of-failure": "[SPOF]",
                "governance-gap": "[GOV ]",
            }.get(vs.status, "[?   ]")
            owner_str = f"  owner: {vs.owner}" if vs.owner else "  owner: UNASSIGNED"
            lines.append(f"  {status_tag} {vs.vector_key:<30}{owner_str}")
            if vs.gap_reason:
                lines.append(f"           -> {vs.gap_reason}")
        lines.append("-" * 64)
        lines.append(
            f"  SPOFs: {self.single_point_of_failure_count()} | "
            f"Gaps: {self.gap_count()} | "
            f"Governance gaps: {self.governance_gap_count()}"
        )
        lines.append("=" * 64)
        return "\n".join(lines)


def generate_exposure_map(
    deployment_type: _DeploymentType,
    team_roles: "TeamRoles",
    pii_in_context: bool = False,
    uses_rag: bool = False,
    has_agent: bool = False,
) -> ExposureMap:
    """
    Generate a structured exposure map for an LLM deployment.

    Infers coverage status from the deployment configuration and team roster.
    Flags gaps, single-points-of-failure, and governance gaps automatically.

    Parameters
    ----------
    deployment_type:
        "chat" | "rag" | "agent" | "api"
    team_roles:
        TeamRoles instance with assigned role names.
    pii_in_context:
        True if user PII can appear in model inputs or outputs.
    uses_rag:
        True if the system retrieves documents into context.
    has_agent:
        True if the system performs multi-step tool-calling.

    Returns
    -------
    ExposureMap with per-vector status and a printable summary.
    """
    vectors: list[ExposureVectorStatus] = []

    # --- Hallucination containment ---
    if team_roles.ml_engineer is None:
        status: _CoverageStatus = "gap"
        reason = "No ML engineer assigned. Hallucination evaluation pipeline unowned."
    else:
        status = "covered"
        reason = ""
    vectors.append(ExposureVectorStatus(
        vector_key="hallucination_containment",
        description=_VECTOR_DESCRIPTIONS["hallucination_containment"],
        status=status,
        owner=team_roles.ml_engineer,
        gap_reason=reason,
    ))

    # --- Adversarial hardening ---
    if team_roles.security_engineer is None:
        status = "gap"
        reason = "No security engineer assigned. Red-teaming and injection defense unowned."
    else:
        status = "covered"
        reason = ""
    vectors.append(ExposureVectorStatus(
        vector_key="adversarial_hardening",
        description=_VECTOR_DESCRIPTIONS["adversarial_hardening"],
        status=status,
        owner=team_roles.security_engineer,
        gap_reason=reason,
    ))

    # --- Agentic safety ---
    if has_agent or deployment_type == "agent":
        if team_roles.platform_engineer is None:
            status = "gap"
            reason = "Agent deployment with no platform engineer. Execution-layer scope enforcement unowned."
        else:
            status = "covered"
            reason = ""
    else:
        status = "covered"
        reason = ""
    vectors.append(ExposureVectorStatus(
        vector_key="agentic_safety",
        description=_VECTOR_DESCRIPTIONS["agentic_safety"],
        status=status,
        owner=team_roles.platform_engineer,
        gap_reason=reason,
    ))

    # --- Data leakage prevention ---
    if pii_in_context or uses_rag or deployment_type == "rag":
        if team_roles.data_engineer is None:
            status = "gap"
            reason = "PII in context or RAG in use but no data engineer assigned."
        elif team_roles.ml_engineer and team_roles.security_engineer:
            # Both ML and security share output filtering — potential governance gap
            status = "governance-gap"
            reason = (
                f"Both '{team_roles.ml_engineer}' (ML) and "
                f"'{team_roles.security_engineer}' (Security) share output filtering "
                "without a documented decision protocol."
            )
        else:
            status = "covered"
            reason = ""
    else:
        status = "covered"
        reason = ""
    vectors.append(ExposureVectorStatus(
        vector_key="data_leakage_prevention",
        description=_VECTOR_DESCRIPTIONS["data_leakage_prevention"],
        status=status,
        owner=team_roles.data_engineer,
        gap_reason=reason,
    ))

    # --- Compliance readiness ---
    if team_roles.compliance_lead is None:
        status = "gap"
        reason = "No compliance lead assigned. Annex IV artifact production unowned."
    elif team_roles.platform_engineer is None:
        status = "single-point-of-failure"
        reason = (
            f"Compliance requires CI/CD artifacts, but platform engineer is unassigned. "
            f"'{team_roles.compliance_lead}' is a single point of failure."
        )
    else:
        status = "covered"
        reason = ""
    vectors.append(ExposureVectorStatus(
        vector_key="compliance_readiness",
        description=_VECTOR_DESCRIPTIONS["compliance_readiness"],
        status=status,
        owner=team_roles.compliance_lead,
        gap_reason=reason,
    ))

    return ExposureMap(
        deployment_type=deployment_type,
        team_roles=team_roles,
        vectors=vectors,
    )


# ---------------------------------------------------------------------------
# Listing 1.6: HardeningReadinessReport — versioned machine-readable report
# Requirements: dataclasses (stdlib), json (stdlib), datetime (stdlib)
# ---------------------------------------------------------------------------

import json as _json
import os as _os
from dataclasses import asdict as _asdict
from datetime import datetime as _dt, timezone as _tz


@_dataclass
class VectorResult:
    """Per-vector score and gap summary for the versioned readiness report."""
    vector_key: str
    label: str
    score: int
    max_score: int
    chapter_reference: str
    gaps: list[str] = _field(default_factory=list)

    @property
    def score_pct(self) -> float:
        """Score as a percentage of the maximum."""
        return round(self.score / self.max_score * 100, 1) if self.max_score else 0.0


@_dataclass
class HardeningReadinessReport:
    """
    Versioned, machine-readable hardening readiness report (Listing 1.6, section 1.7.5).

    Wraps a completed HardeningReport (Listing 1.5) into a JSON artifact that
    a CI system can parse, compare over time, and include in an Annex IV
    technical file.  Each run produces a new file with the deployment ID
    and date in the filename, so the history is the audit trail.

    Usage
    -----
    >>> report = HardeningReport.score_from_dict({"HC1": True, "HC2": False, ...})
    >>> readiness = HardeningReadinessReport.from_report(
    ...     report,
    ...     deployment_id="customer-rag-chatbot-prod-v2",
    ... )
    >>> readiness.save("/tmp/")
    """

    deployment_id: str
    assessor: str
    assessment_timestamp: str = _field(
        default_factory=lambda: _dt.now(_tz.utc).isoformat()
    )
    total_score: int = 0
    max_score: int = 25
    exposure_tier: str = "CRITICAL"      # "CRITICAL" | "PARTIAL" | "DEFENSIBLE"
    vectors: list[VectorResult] = _field(default_factory=list)
    recommended_chapters: list[str] = _field(default_factory=list)
    answers: dict[str, bool] = _field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_report(
        cls,
        report: "HardeningReport",
        deployment_id: str,
    ) -> "HardeningReadinessReport":
        """
        Build a HardeningReadinessReport directly from a completed HardeningReport.

        Consumes the HardeningReport's own vector_results() and exposure_tier
        (Listing 1.5) rather than re-deriving a second, parallel classification
        from the raw answers dict — the two reports must always agree, since
        this one is built from the other's own output, not a fresh computation.

        Parameters
        ----------
        report:
            A HardeningReport instance with all 25 questions answered.
        deployment_id:
            Human-readable identifier for the deployment, e.g.
            "customer-rag-chatbot-prod-v2".
        """
        vectors = [
            VectorResult(
                vector_key=v["key"],
                label=v["label"],
                score=v["score"],
                max_score=v["max_score"],
                chapter_reference=v["chapter_reference"],
                gaps=v["gaps"],
            )
            for v in report.vector_results()
        ]

        recommended = [v.chapter_reference for v in vectors if v.score < 3]

        return cls(
            deployment_id=deployment_id,
            assessor=report.assessor,
            total_score=report.total_score,
            max_score=report.max_score,
            exposure_tier=report.exposure_tier.name,
            vectors=vectors,
            recommended_chapters=recommended,
            answers=dict(report.answers),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "assessor": self.assessor,
            "assessment_timestamp": self.assessment_timestamp,
            "total_score": self.total_score,
            "max_score": self.max_score,
            "score_pct": round(self.total_score / self.max_score * 100, 1),
            "exposure_tier": self.exposure_tier,
            "vectors": [
                {
                    "vector_key": v.vector_key,
                    "label": v.label,
                    "score": v.score,
                    "max_score": v.max_score,
                    "score_pct": v.score_pct,
                    "chapter_reference": v.chapter_reference,
                    "gaps": v.gaps,
                }
                for v in self.vectors
            ],
            "recommended_chapters": self.recommended_chapters,
            "answers": self.answers,
        }

    def save(self, output_dir: str = ".") -> str:
        """
        Save the report to a versioned JSON file.

        Filename format: hardening-report-{deployment_id}-{date}.json
        Returns the full file path.
        """
        date_str = self.assessment_timestamp[:10]  # YYYY-MM-DD
        slug = self.deployment_id.lower().replace(" ", "-").replace("/", "-")
        filename = f"hardening-report-{slug}-{date_str}.json"
        path = _os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(self.to_dict(), fh, indent=2)
        print(f"[HardeningReadinessReport] Saved to: {path}")
        return path
