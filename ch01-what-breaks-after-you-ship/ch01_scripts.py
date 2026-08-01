"""
Chapter 1: What Breaks After You Ship
Hardening LLM Systems in Production — Companion Code
Author: Rudrendu Paul | https://orcid.org/0009-0008-0141-4690

This module implements the HardeningReport self-diagnostic scorecard: 25 yes/no questions
across 5 hardening vectors. Run it against any LLM-powered system to get a readiness score
and a prioritized remediation plan before incidents occur in production.

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
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

HARDENING_VECTORS = [
    "Hallucination & Factuality",
    "Prompt Injection & Adversarial Inputs",
    "Output Safety & Policy Compliance",
    "Observability & Incident Response",
    "Governance & Access Control",
]

# 25 questions: 5 per vector, in the same order as HARDENING_VECTORS
DIAGNOSTIC_QUESTIONS: list[dict] = [
    # --- Vector 1: Hallucination & Factuality ---
    {
        "id": "H1",
        "vector": "Hallucination & Factuality",
        "question": "Do you run an automated faithfulness metric (e.g. RAGAS, deepeval) on every model response in CI?",
        "rationale": "Without an automated gate, hallucination regressions ship silently.",
        "remediation": "Add a faithfulness score threshold to your CI pipeline. Chapter 2 covers deepeval + RAGAS setup.",
    },
    {
        "id": "H2",
        "vector": "Hallucination & Factuality",
        "question": "Do you track hallucination rate as a production KPI on a dashboard?",
        "rationale": "You cannot improve what you do not measure in the live environment.",
        "remediation": "Instrument your response pipeline to log faithfulness scores; route them to Grafana or Datadog.",
    },
    {
        "id": "H3",
        "vector": "Hallucination & Factuality",
        "question": "Do you run self-consistency checks (sample N responses, compare) for high-stakes outputs?",
        "rationale": "A single forward pass is insufficient evidence that a claim is reliable.",
        "remediation": "Use the SelfConsistencyChecker from Chapter 3 on any output that triggers a real-world action.",
    },
    {
        "id": "H4",
        "vector": "Hallucination & Factuality",
        "question": "Do you decompose compound claims into atomic assertions before scoring?",
        "rationale": "Composite answers hide per-claim accuracy. A response can be 80% correct and catastrophically wrong.",
        "remediation": "Implement the ClaimDecompositionPipeline from Chapter 3.",
    },
    {
        "id": "H5",
        "vector": "Hallucination & Factuality",
        "question": "Do you maintain a regression test suite of known hallucination cases?",
        "rationale": "Fixed hallucinations re-emerge after model updates or prompt changes without a regression suite.",
        "remediation": "Collect every production hallucination into a golden test set. Run it on every deployment.",
    },
    # --- Vector 2: Prompt Injection & Adversarial Inputs ---
    {
        "id": "P1",
        "vector": "Prompt Injection & Adversarial Inputs",
        "question": "Do you scan user inputs for direct prompt injection patterns before they reach the model?",
        "rationale": "User-controlled text entering the system prompt is the most common attack surface.",
        "remediation": "Add an injection classifier layer. Chapter 4 covers detection approaches and open-source libraries.",
    },
    {
        "id": "P2",
        "vector": "Prompt Injection & Adversarial Inputs",
        "question": "Do you sanitize or constrain tool-call arguments returned by the model before execution?",
        "rationale": "Models can be induced to emit malicious tool arguments. Validation is mandatory before execution.",
        "remediation": "Parse and validate all tool call payloads against a strict schema before invoking downstream APIs.",
    },
    {
        "id": "P3",
        "vector": "Prompt Injection & Adversarial Inputs",
        "question": "Do you apply structural separation between system instructions and user content in the prompt?",
        "rationale": "Embedding user content inside the system prompt makes injection trivially easy.",
        "remediation": "Use the multi-turn message format with explicit role boundaries; never f-string user input into system prompts.",
    },
    {
        "id": "P4",
        "vector": "Prompt Injection & Adversarial Inputs",
        "question": "Do you red-team your prompts quarterly for jailbreak and indirect injection vulnerabilities?",
        "rationale": "Prompt injection attack techniques evolve continuously; static defenses decay.",
        "remediation": "Schedule a red-team sprint using the framework in Chapter 6.",
    },
    {
        "id": "P5",
        "vector": "Prompt Injection & Adversarial Inputs",
        "question": "Do you monitor for anomalous token patterns or unusually long inputs in production traffic?",
        "rationale": "Many injection attempts include unusual prompt structures detectable by statistical monitoring.",
        "remediation": "Add input-length percentile alerts and token-pattern anomaly detection to your ingress layer.",
    },
    # --- Vector 3: Output Safety & Policy Compliance ---
    {
        "id": "O1",
        "vector": "Output Safety & Policy Compliance",
        "question": "Do you run outputs through a policy classifier (e.g. LlamaGuard or custom classifier) before returning to users?",
        "rationale": "Base models and fine-tunes produce policy-violating outputs under adversarial and benign inputs.",
        "remediation": "Wrap model inference with a classifier layer. Set up alerts for any policy violation above threshold.",
    },
    {
        "id": "O2",
        "vector": "Output Safety & Policy Compliance",
        "question": "Do you have a documented and tested fallback response for every safety refusal category?",
        "rationale": "Unclear or overly terse refusals damage user experience and invite bypass attempts.",
        "remediation": "Write explicit fallback copy for each refusal category. A/B test fallback acceptance rates.",
    },
    {
        "id": "O3",
        "vector": "Output Safety & Policy Compliance",
        "question": "Do you scan outputs for PII (names, email, phone, SSN) before returning them to clients?",
        "rationale": "Models memorize and reproduce training data. PII leakage is both a compliance and a trust failure.",
        "remediation": "Deploy a PII detector in the output path. Chapter 9 covers detection and redaction pipelines.",
    },
    {
        "id": "O4",
        "vector": "Output Safety & Policy Compliance",
        "question": "Do you store output safety violation logs for audit and model improvement?",
        "rationale": "Violations logged only as metrics cannot be used for targeted retraining or policy refinement.",
        "remediation": "Log the full prompt-response pair (with user consent where required) to a secured audit store.",
    },
    {
        "id": "O5",
        "vector": "Output Safety & Policy Compliance",
        "question": "Do you have a circuit-breaker that halts model traffic if violation rate exceeds a threshold?",
        "rationale": "Safety regressions introduced by model updates can spike violation rates before any human reviews logs.",
        "remediation": "Implement a sliding-window violation rate alarm with automated traffic kill-switch capability.",
    },
    # --- Vector 4: Observability & Incident Response ---
    {
        "id": "I1",
        "vector": "Observability & Incident Response",
        "question": "Do you emit structured logs (prompt, response, latency, model version, user_id) for every LLM call?",
        "rationale": "Unstructured logs make post-incident forensics slow and often impossible.",
        "remediation": "Add a structured logging wrapper to every model call. Include a trace_id for distributed tracing.",
    },
    {
        "id": "I2",
        "vector": "Observability & Incident Response",
        "question": "Do you have a runbook for LLM incidents (hallucination spike, injection detected, PII leak)?",
        "rationale": "Incident response improvised under pressure is slower and makes more mistakes than a rehearsed runbook.",
        "remediation": "Write and drill a runbook for your top 3 incident types. Chapter 8 provides a template.",
    },
    {
        "id": "I3",
        "vector": "Observability & Incident Response",
        "question": "Do you track model latency p50, p95, and p99 and alert on degradation?",
        "rationale": "Latency spikes often precede or accompany quality degradation, especially under load.",
        "remediation": "Add latency percentile dashboards and SLO-linked alerts.",
    },
    {
        "id": "I4",
        "vector": "Observability & Incident Response",
        "question": "Do you use shadow traffic to test new model versions against production traffic before cutover?",
        "rationale": "Lab evaluations miss the long tail of production inputs that trigger edge-case failures.",
        "remediation": "Implement the ShadowTrafficHarness from Chapter 3 to mirror a percentage of live traffic to candidate models.",
    },
    {
        "id": "I5",
        "vector": "Observability & Incident Response",
        "question": "Do you have a rollback procedure (tested) to revert to a prior model version within 15 minutes?",
        "rationale": "Novel model versions can degrade quality or safety suddenly; rollback speed is a critical SLO.",
        "remediation": "Version your model endpoints and test the rollback procedure quarterly.",
    },
    # --- Vector 5: Governance & Access Control ---
    {
        "id": "G1",
        "vector": "Governance & Access Control",
        "question": "Do you maintain a model registry that records which model version is serving each endpoint in production?",
        "rationale": "Without a registry, incidents cannot be correlated with model versions, making root cause analysis guesswork.",
        "remediation": "Adopt a lightweight model registry (e.g. MLflow, DVC, or a version tag in your config store).",
    },
    {
        "id": "G2",
        "vector": "Governance & Access Control",
        "question": "Do you apply least-privilege access controls so the model can only call the tools it needs for each task?",
        "rationale": "Overly permissive tool grants amplify the blast radius of a prompt injection or model error.",
        "remediation": "Define per-task tool allow-lists. Audit tool grants quarterly. Chapter 7 covers agentic scope containment.",
    },
    {
        "id": "G3",
        "vector": "Governance & Access Control",
        "question": "Do you document model cards or system cards for every model in production?",
        "rationale": "Model cards are required for EU AI Act compliance and are the baseline for responsible deployment.",
        "remediation": "Write a model card for each production model. Chapter 10 provides an EU AI Act aligned template.",
    },
    {
        "id": "G4",
        "vector": "Governance & Access Control",
        "question": "Do you review and approve prompt changes through a code review process before production deployment?",
        "rationale": "Prompt changes are code changes. Unreviewed prompt edits have caused production safety incidents.",
        "remediation": "Store system prompts in version control. Apply PR-based review gates. Chapter 11 covers the full PR gate.",
    },
    {
        "id": "G5",
        "vector": "Governance & Access Control",
        "question": "Do you have a designated AI risk owner (a person, not a team) accountable for production LLM incidents?",
        "rationale": "Diffuse accountability means no one owns the incident until it is too late.",
        "remediation": "Assign an AI risk owner in your incident response org chart. Include them in all production change reviews.",
    },
]


# ---------------------------------------------------------------------------
# Core scorecard class
# ---------------------------------------------------------------------------

@dataclass
class VectorScore:
    name: str
    answered_yes: int = 0
    total: int = 0
    failed_questions: list[dict] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.answered_yes / self.total if self.total > 0 else 0.0

    @property
    def score_pct(self) -> int:
        return round(self.score * 100)

    @property
    def risk_level(self) -> str:
        if self.score >= 0.8:
            return "LOW"
        if self.score >= 0.6:
            return "MEDIUM"
        if self.score >= 0.4:
            return "HIGH"
        return "CRITICAL"

    @property
    def risk_emoji(self) -> str:
        mapping = {"LOW": "[OK]", "MEDIUM": "[WARN]", "HIGH": "[HIGH]", "CRITICAL": "[CRIT]"}
        return mapping[self.risk_level]


@dataclass
class HardeningReport:
    """
    Self-diagnostic scorecard for LLM systems in production.

    Covers 25 questions across 5 hardening vectors. Each question is answered
    yes (1) or no (0). The overall score drives a readiness band and a
    prioritized remediation plan targeting the most dangerous gaps first.

    Usage
    -----
    >>> report = HardeningReport()
    >>> report.answer("H1", True)
    >>> report.answer("H2", False)
    >>> print(report.summary())

    Or run the full interactive diagnostic:
    >>> report = HardeningReport.run_interactive()
    """

    answers: dict[str, bool] = field(default_factory=dict)
    system_name: str = "My LLM System"
    assessor: str = "Unknown"

    # ------------------------------------------------------------------
    # Answering questions
    # ------------------------------------------------------------------

    def answer(self, question_id: str, response: bool) -> None:
        """Record a yes/no answer for a question ID (e.g. 'H1', 'P3')."""
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

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_vector_scores(self) -> dict[str, VectorScore]:
        scores: dict[str, VectorScore] = {v: VectorScore(name=v) for v in HARDENING_VECTORS}
        for q in DIAGNOSTIC_QUESTIONS:
            vector_score = scores[q["vector"]]
            vector_score.total += 1
            answered = self.answers.get(q["id"])
            if answered is True:
                vector_score.answered_yes += 1
            elif answered is False:
                vector_score.failed_questions.append(q)
        return scores

    @property
    def total_score(self) -> float:
        answered_yes = sum(1 for v in self.answers.values() if v)
        total = len(DIAGNOSTIC_QUESTIONS)
        return answered_yes / total if total > 0 else 0.0

    @property
    def total_score_pct(self) -> int:
        return round(self.total_score * 100)

    @property
    def readiness_band(self) -> str:
        if self.total_score >= 0.80:
            return "HARDENED"
        if self.total_score >= 0.60:
            return "REINFORCED"
        if self.total_score >= 0.40:
            return "EXPOSED"
        return "VULNERABLE"

    @property
    def answered_count(self) -> int:
        return len(self.answers)

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
        lines.append(f"  OVERALL SCORE : {self.total_score_pct}% ({self.answered_count}/25 questions answered)")
        lines.append(f"  READINESS BAND: {self.readiness_band}")
        lines.append("")
        lines.append("-" * 68)
        lines.append("  VECTOR SCORES")
        lines.append("-" * 68)

        for vs in vector_scores.values():
            bar = _progress_bar(vs.score, width=20)
            lines.append(f"  {vs.risk_emoji} {vs.name:<40} {bar} {vs.score_pct:>3}%  [{vs.risk_level}]")

        lines.append("")
        lines.append("-" * 68)
        lines.append("  PRIORITIZED REMEDIATION PLAN")
        lines.append("  (ordered by vector risk, highest first)")
        lines.append("-" * 68)

        # Sort vectors by ascending score (worst first)
        sorted_vectors = sorted(vector_scores.values(), key=lambda vs: vs.score)
        item_num = 1
        for vs in sorted_vectors:
            if not vs.failed_questions:
                continue
            lines.append(f"\n  {vs.risk_emoji} {vs.name.upper()} — {vs.score_pct}% ({vs.risk_level} risk)")
            for q in vs.failed_questions:
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
        vector_scores = self._compute_vector_scores()
        return {
            "system_name": self.system_name,
            "assessor": self.assessor,
            "date": _today(),
            "overall_score_pct": self.total_score_pct,
            "readiness_band": self.readiness_band,
            "answered_questions": self.answered_count,
            "total_questions": len(DIAGNOSTIC_QUESTIONS),
            "vectors": [
                {
                    "name": vs.name,
                    "score_pct": vs.score_pct,
                    "risk_level": vs.risk_level,
                    "answered_yes": vs.answered_yes,
                    "total": vs.total,
                    "gaps": [q["id"] for q in vs.failed_questions],
                }
                for vs in vector_scores.values()
            ],
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
                print(f"\n  == {current_vector.upper()} ==")

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
        "mixed" — realistic mid-stage team (default)
        "best"  — all controls in place
        "worst" — no controls in place
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

    # Mixed scenario: a realistic early-production LLM system
    mixed_answers: dict[str, bool] = {
        # Hallucination: CI metric in place, dashboard not yet; no self-consistency
        "H1": True,  "H2": False, "H3": False, "H4": False, "H5": True,
        # Prompt injection: input scan exists, tool-arg validation missing
        "P1": True,  "P2": False, "P3": True,  "P4": False, "P5": False,
        # Output safety: classifier deployed, no PII scan, no circuit-breaker
        "O1": True,  "O2": True,  "O3": False, "O4": True,  "O5": False,
        # Observability: structured logs exist, no runbook, no shadow traffic
        "I1": True,  "I2": False, "I3": True,  "I4": False, "I5": False,
        # Governance: no model registry, no model card, PR gate in place
        "G1": False, "G2": False, "G3": False, "G4": True,  "G5": False,
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
    Versioned, machine-readable hardening readiness report.

    Wraps scorecard results from HardeningReport into a JSON artifact that
    a CI system can parse, compare over time, and include in an Annex IV
    technical file.  Each run produces a new file with the deployment ID
    and date in the filename, so the history is the audit trail.

    Usage
    -----
    >>> answers = {"H1": True, "H2": False, ...}  # from HardeningReport
    >>> report = HardeningReadinessReport.from_answers(
    ...     answers=answers,
    ...     deployment_id="customer-rag-chatbot-prod-v2",
    ...     assessor="Jane Smith",
    ... )
    >>> report.save("/tmp/")
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
    def from_answers(
        cls,
        answers: dict[str, bool],
        deployment_id: str,
        assessor: str = "Unknown",
    ) -> "HardeningReadinessReport":
        """
        Build a HardeningReadinessReport from a completed HardeningReport answer dict.

        Parameters
        ----------
        answers:
            dict mapping question IDs (e.g. "H1") to bool.
            Typically from HardeningReport.answers after running the diagnostic.
        deployment_id:
            Human-readable identifier for the deployment, e.g.
            "customer-rag-chatbot-prod-v2".
        assessor:
            Name of the person who completed the diagnostic.
        """
        # Map question IDs to vectors and labels
        _VECTOR_MAP = {
            "H": ("hallucination_factuality", "Hallucination & Factuality", "Chapters 2–3"),
            "P": ("prompt_injection", "Prompt Injection & Adversarial Inputs", "Chapters 4–6"),
            "O": ("output_safety", "Output Safety & Policy Compliance", "Chapter 9"),
            "I": ("observability", "Observability & Incident Response", "Chapters 7, 11"),
            "G": ("governance", "Governance & Access Control", "Chapters 10–11"),
        }

        vector_scores: dict[str, dict] = {}
        for prefix, (key, label, chapter) in _VECTOR_MAP.items():
            qs = [qid for qid in answers if qid.startswith(prefix)]
            yes_count = sum(1 for qid in qs if answers.get(qid))
            gaps = [qid for qid in qs if not answers.get(qid)]
            vector_scores[prefix] = {
                "key": key, "label": label, "chapter": chapter,
                "score": yes_count, "max": len(qs) if qs else 5, "gaps": gaps,
            }

        total = sum(v["score"] for v in vector_scores.values())
        exposure = (
            "DEFENSIBLE" if total >= 19
            else "PARTIAL" if total >= 11
            else "CRITICAL"
        )

        recommended = [
            v["chapter"]
            for v in vector_scores.values()
            if v["score"] < 3
        ]

        vectors = [
            VectorResult(
                vector_key=v["key"],
                label=v["label"],
                score=v["score"],
                max_score=v["max"],
                chapter_reference=v["chapter"],
                gaps=v["gaps"],
            )
            for v in vector_scores.values()
        ]

        return cls(
            deployment_id=deployment_id,
            assessor=assessor,
            total_score=total,
            exposure_tier=exposure,
            vectors=vectors,
            recommended_chapters=recommended,
            answers=answers,
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
