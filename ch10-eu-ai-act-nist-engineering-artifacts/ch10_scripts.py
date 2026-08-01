"""
Chapter 11 — EU AI Act + NIST AI RMF Compliance Automation
===========================================================
Manning book: "Hardening LLM Systems in Production" by Rudrendu Paul

Companion script covering:
  - AnnexIVPackage dataclass with completeness_score()
  - Annex IV CI gate (sys.exit(1) on failure)
  - Output provenance recorder with HMAC-SHA256 signature
  - TamperEvidentAuditLog with chained-hash tamper detection
  - NISTAI6001Tracker with ImplementationStatus enum and gap_report()
  - Dual-framework mapping report (EU AI Act + NIST cross-reference)
  - PostMarketMonitoringReport dataclass

Dependencies: pyyaml>=6.0  (all others are stdlib)
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # pyyaml>=6.0


# ---------------------------------------------------------------------------
# 1. AnnexIVPackage — EU AI Act Article 11 / Annex IV documentation bundle
# ---------------------------------------------------------------------------

ANNEX_IV_REQUIRED_FIELDS: List[str] = [
    "system_name",
    "system_version",
    "intended_purpose",
    "risk_category",          # Article 6: limited / high / unacceptable
    "provider_name",
    "provider_address",
    "contact_email",
    "general_description",    # Annex IV §1
    "design_specifications",  # Annex IV §2
    "training_data_summary",  # Annex IV §3
    "validation_testing",     # Annex IV §4
    "technical_standards",    # Annex IV §5
    "post_market_plan",       # Annex IV §6
    "human_oversight_measures",  # Article 14
    "accuracy_metrics",
    "robustness_measures",
    "cybersecurity_measures",  # Annex IV §7
    "declaration_of_conformity",
]

ANNEX_IV_OPTIONAL_FIELDS: List[str] = [
    "eu_database_registration_id",
    "notified_body_id",
    "third_party_audit_report",
    "bias_assessment_report",
    "explainability_documentation",
]


@dataclass
class AnnexIVPackage:
    """
    Structured representation of an EU AI Act Annex IV technical documentation
    bundle for a high-risk AI system.

    All required fields must be non-empty strings for completeness_score == 1.0.
    Optional fields raise the score beyond the minimum-pass threshold.
    """

    # --- Required fields ---
    system_name: str = ""
    system_version: str = ""
    intended_purpose: str = ""
    risk_category: str = ""
    provider_name: str = ""
    provider_address: str = ""
    contact_email: str = ""
    general_description: str = ""
    design_specifications: str = ""
    training_data_summary: str = ""
    validation_testing: str = ""
    technical_standards: str = ""
    post_market_plan: str = ""
    human_oversight_measures: str = ""
    accuracy_metrics: str = ""
    robustness_measures: str = ""
    cybersecurity_measures: str = ""
    declaration_of_conformity: str = ""

    # --- Optional fields ---
    eu_database_registration_id: str = ""
    notified_body_id: str = ""
    third_party_audit_report: str = ""
    bias_assessment_report: str = ""
    explainability_documentation: str = ""

    # --- Metadata ---
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "annex-iv-v1.0"

    def completeness_score(self) -> float:
        """
        Return a float in [0.0, 1.0] representing how complete this package is.

        Formula:
          required_weight = 0.85  (all 18 required fields, equally weighted)
          optional_weight = 0.15  (5 optional fields, equally weighted)
        A score >= 0.85 means all required fields are present (CI gate threshold).
        """
        req_filled = sum(
            1 for f in ANNEX_IV_REQUIRED_FIELDS if getattr(self, f, "").strip()
        )
        opt_filled = sum(
            1 for f in ANNEX_IV_OPTIONAL_FIELDS if getattr(self, f, "").strip()
        )
        req_score = (req_filled / len(ANNEX_IV_REQUIRED_FIELDS)) * 0.85
        opt_score = (opt_filled / len(ANNEX_IV_OPTIONAL_FIELDS)) * 0.15
        return round(req_score + opt_score, 4)

    def missing_required_fields(self) -> List[str]:
        """Return list of required fields that are empty or whitespace-only."""
        return [f for f in ANNEX_IV_REQUIRED_FIELDS if not getattr(self, f, "").strip()]

    def to_yaml(self) -> str:
        """Serialize package to a YAML string for storage / version control."""
        return yaml.dump(asdict(self), default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "AnnexIVPackage":
        """Deserialize from a YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# 2. Annex IV CI Gate
# ---------------------------------------------------------------------------

ANNEX_IV_CI_THRESHOLD = 0.85  # All required fields must be present


def run_annex_iv_ci_gate(package: AnnexIVPackage, *, strict: bool = True) -> None:
    """
    CI/CD gate that checks Annex IV documentation completeness.

    Parameters
    ----------
    package : AnnexIVPackage
        The documentation bundle to validate.
    strict : bool
        If True (default), call sys.exit(1) on failure so CI pipelines catch it.
        If False, raise ValueError instead (useful in test suites).

    Exit codes
    ----------
    0 — all required fields present; score >= threshold
    1 — one or more required fields missing
    """
    score = package.completeness_score()
    missing = package.missing_required_fields()

    print(f"[Annex IV Gate] System: {package.system_name!r} v{package.system_version}")
    print(f"[Annex IV Gate] Completeness score: {score:.2%} (threshold: {ANNEX_IV_CI_THRESHOLD:.2%})")

    if missing:
        print("[Annex IV Gate] FAIL — missing required fields:")
        for field_name in missing:
            print(f"  - {field_name}")
        if strict:
            sys.exit(1)
        raise ValueError(f"Annex IV incomplete: {missing}")

    print("[Annex IV Gate] PASS — all required Annex IV fields present.")


# ---------------------------------------------------------------------------
# 3. Output Provenance Recorder (HMAC-SHA256)
# ---------------------------------------------------------------------------

def _get_hmac_key() -> bytes:
    """
    Retrieve the HMAC signing key from the environment.
    In production, inject via a secrets manager (Vault, AWS Secrets Manager, etc.).
    """
    key_hex = os.environ.get("LLM_PROVENANCE_HMAC_KEY", "")
    if not key_hex:
        # Fallback for local dev — NOT for production
        key_hex = "deadbeef" * 8  # 32-byte all-zeros equivalent placeholder
    return bytes.fromhex(key_hex)


def sign_output(payload: Dict[str, Any]) -> str:
    """
    Compute an HMAC-SHA256 signature over the canonical JSON representation
    of a provenance payload.

    Returns the hex-encoded signature string.
    """
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    sig = hmac.new(_get_hmac_key(), canonical.encode("utf-8"), hashlib.sha256)
    return sig.hexdigest()


@dataclass
class ProvenanceRecord:
    """
    A tamper-evident record linking an LLM output to its input context,
    model identity, and runtime parameters.
    """
    record_id: str
    model_id: str
    model_version: str
    prompt_hash: str        # SHA-256 of the full prompt (hex)
    output_hash: str        # SHA-256 of the raw output (hex)
    timestamp_utc: str
    temperature: float
    max_tokens: int
    system_prompt_version: str
    user_id: str
    session_id: str
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "record_id": self.record_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "prompt_hash": self.prompt_hash,
            "output_hash": self.output_hash,
            "timestamp_utc": self.timestamp_utc,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt_version": self.system_prompt_version,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        self.signature = sign_output(payload)

    def verify(self) -> bool:
        """Re-derive the signature and compare; returns True if record is intact."""
        payload = {
            "record_id": self.record_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "prompt_hash": self.prompt_hash,
            "output_hash": self.output_hash,
            "timestamp_utc": self.timestamp_utc,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt_version": self.system_prompt_version,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        expected = sign_output(payload)
        return hmac.compare_digest(expected, self.signature)


class OutputProvenanceRecorder:
    """
    Records and persists provenance records for every LLM output.

    Usage
    -----
    recorder = OutputProvenanceRecorder(log_path=Path("provenance.jsonl"))
    record = recorder.record(
        prompt="What is the capital of France?",
        output="Paris.",
        model_id="gpt-4o",
        model_version="2024-08-06",
        temperature=0.0,
        max_tokens=256,
        system_prompt_version="v3.1",
        user_id="u-001",
        session_id="sess-xyz",
    )
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def record(
        self,
        prompt: str,
        output: str,
        model_id: str,
        model_version: str,
        temperature: float,
        max_tokens: int,
        system_prompt_version: str,
        user_id: str,
        session_id: str,
    ) -> ProvenanceRecord:
        import uuid

        rec = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            model_id=model_id,
            model_version=model_version,
            prompt_hash=self._sha256(prompt),
            output_hash=self._sha256(output),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt_version=system_prompt_version,
            user_id=user_id,
            session_id=session_id,
        )
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")
        return rec

    def load_all(self) -> List[ProvenanceRecord]:
        records = []
        if not self.log_path.exists():
            return records
        with open(self.log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Rebuild without triggering __post_init__ signature re-derivation
                rec = ProvenanceRecord.__new__(ProvenanceRecord)
                rec.__dict__.update(data)
                records.append(rec)
        return records


# ---------------------------------------------------------------------------
# 4. TamperEvidentAuditLog — chained-hash append-only log
# ---------------------------------------------------------------------------

class TamperEvidentAuditLog:
    """
    An append-only audit log where each entry's hash chains to the previous,
    making any deletion or modification of a historical record detectable.

    Each log entry is a JSON line with fields:
      seq, timestamp_utc, event_type, payload, prev_hash, entry_hash

    entry_hash = SHA-256(seq + timestamp_utc + event_type + payload_json + prev_hash)
    """

    GENESIS_HASH = "0" * 64  # Sentinel for the first entry

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash: str = self.GENESIS_HASH
        self._seq: int = 0
        self._bootstrap()

    def _bootstrap(self) -> None:
        """Replay existing log to restore last_hash and seq counter."""
        if not self.log_path.exists():
            return
        with open(self.log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._last_hash = entry["entry_hash"]
                self._seq = entry["seq"]

    def _compute_entry_hash(
        self, seq: int, timestamp: str, event_type: str, payload_json: str, prev_hash: str
    ) -> str:
        raw = f"{seq}{timestamp}{event_type}{payload_json}{prev_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def append(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Append a new entry to the log; returns the full entry dict."""
        self._seq += 1
        timestamp = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload, sort_keys=True)
        entry_hash = self._compute_entry_hash(
            self._seq, timestamp, event_type, payload_json, self._last_hash
        )
        entry = {
            "seq": self._seq,
            "timestamp_utc": timestamp,
            "event_type": event_type,
            "payload": payload,
            "prev_hash": self._last_hash,
            "entry_hash": entry_hash,
        }
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        self._last_hash = entry_hash
        return entry

    def verify_integrity(self) -> bool:
        """
        Replay the entire log and verify the hash chain is unbroken.

        Returns True if no tampering detected; False (with printed diagnostics)
        if any entry's recomputed hash does not match the stored entry_hash,
        or if any entry's prev_hash does not match the preceding entry's entry_hash.
        """
        if not self.log_path.exists():
            print("[AuditLog] No log file found — nothing to verify.")
            return True

        entries = []
        with open(self.log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        if not entries:
            print("[AuditLog] Empty log — integrity OK.")
            return True

        prev_hash = self.GENESIS_HASH
        ok = True

        for entry in entries:
            seq = entry["seq"]
            expected_hash = self._compute_entry_hash(
                seq,
                entry["timestamp_utc"],
                entry["event_type"],
                json.dumps(entry["payload"], sort_keys=True),
                entry["prev_hash"],
            )
            if entry["entry_hash"] != expected_hash:
                print(f"[AuditLog] TAMPER DETECTED at seq={seq}: entry_hash mismatch")
                ok = False
            if entry["prev_hash"] != prev_hash:
                print(f"[AuditLog] TAMPER DETECTED at seq={seq}: prev_hash chain broken")
                ok = False
            prev_hash = entry["entry_hash"]

        if ok:
            print(f"[AuditLog] Integrity OK — {len(entries)} entries verified.")
        return ok


# ---------------------------------------------------------------------------
# 5. NIST AI RMF 600-1 Tracker
# ---------------------------------------------------------------------------

class ImplementationStatus(str, Enum):
    """Implementation status for a NIST AI RMF control."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    NOT_APPLICABLE = "not_applicable"


# NIST AI 600-1 Gen AI Profile subcategory codes (illustrative subset)
NIST_AI_600_1_SUBCATEGORIES: Dict[str, str] = {
    "GV-1.1": "Policies, processes, procedures and practices for organizational TEVV",
    "GV-1.2": "Organizational teams are committed to governance of AI risk",
    "GV-2.1": "Scientific integrity and TEVV considerations are integrated",
    "GV-3.1": "Organizational risk tolerance for AI is established",
    "GV-4.1": "Organizational teams are committed to accountability",
    "GV-5.1": "Policies and practices are in place for AI worker diversity",
    "GV-6.1": "Policies for third-party entities are established",
    "MP-2.1": "Scientific findings used in AI design are identified",
    "MP-2.2": "TEVV plans include consideration of scientific findings",
    "MP-4.1": "Risks and benefits of an AI system are examined",
    "MS-1.1": "AI system risks are identified and assessed",
    "MS-1.2": "Established metrics are used to measure performance",
    "MS-2.1": "Error characteristics are tested across deployment contexts",
    "MS-2.2": "Design decisions are documented",
    "MS-2.5": "AI system to be deployed in high-risk settings goes through rigorous testing",
    "MS-2.6": "Bias testing is conducted",
    "MS-2.10": "Privacy risk is examined",
    "MS-4.1": "Performance on standardized or widely used benchmarks is documented",
    "MG-2.2": "Mechanisms are in place to assess and adjust",
    "MG-3.1": "Risk treatment plans are documented",
    "MG-4.1": "Post-deployment AI risks and benefits are monitored",
    "MG-4.2": "Evaluations of AI risks are used to make adjustments",
}


@dataclass
class NISTControlEntry:
    """Tracks implementation status and evidence for a single NIST AI 600-1 subcategory."""
    subcategory_code: str
    description: str
    status: ImplementationStatus = ImplementationStatus.NOT_STARTED
    evidence_reference: str = ""
    owner: str = ""
    target_date: str = ""
    notes: str = ""


class NISTAI6001Tracker:
    """
    Tracks implementation status of NIST AI RMF 600-1 generative AI profile
    subcategories for an LLM system.

    Usage
    -----
    tracker = NISTAI6001Tracker(system_name="CustomerCareBot")
    tracker.update("GV-1.1", ImplementationStatus.IMPLEMENTED,
                   evidence_reference="governance-policy-v2.pdf",
                   owner="AI Governance Team")
    report = tracker.gap_report()
    """

    def __init__(self, system_name: str) -> None:
        self.system_name = system_name
        self.controls: Dict[str, NISTControlEntry] = {
            code: NISTControlEntry(subcategory_code=code, description=desc)
            for code, desc in NIST_AI_600_1_SUBCATEGORIES.items()
        }

    def update(
        self,
        subcategory_code: str,
        status: ImplementationStatus,
        evidence_reference: str = "",
        owner: str = "",
        target_date: str = "",
        notes: str = "",
    ) -> None:
        """Update the implementation status of a subcategory."""
        if subcategory_code not in self.controls:
            raise KeyError(f"Unknown NIST AI 600-1 subcategory: {subcategory_code!r}")
        ctrl = self.controls[subcategory_code]
        ctrl.status = status
        ctrl.evidence_reference = evidence_reference
        ctrl.owner = owner
        ctrl.target_date = target_date
        ctrl.notes = notes

    def gap_report(self) -> Dict[str, Any]:
        """
        Generate a structured gap report.

        Returns a dict with:
          - system_name
          - generated_at
          - summary: counts by status
          - gaps: list of entries with status NOT_STARTED or IN_PROGRESS
          - implemented: list of implemented entries
          - not_applicable: list of N/A entries
        """
        by_status: Dict[str, List[Dict]] = {
            s.value: [] for s in ImplementationStatus
        }
        for ctrl in self.controls.values():
            by_status[ctrl.status.value].append(asdict(ctrl))

        return {
            "system_name": self.system_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {s: len(entries) for s, entries in by_status.items()},
            "gaps": (
                by_status[ImplementationStatus.NOT_STARTED.value]
                + by_status[ImplementationStatus.IN_PROGRESS.value]
            ),
            "implemented": by_status[ImplementationStatus.IMPLEMENTED.value],
            "not_applicable": by_status[ImplementationStatus.NOT_APPLICABLE.value],
        }

    def completion_percentage(self) -> float:
        """Percentage of applicable controls that are IMPLEMENTED."""
        applicable = [
            c for c in self.controls.values()
            if c.status != ImplementationStatus.NOT_APPLICABLE
        ]
        if not applicable:
            return 0.0
        implemented = sum(
            1 for c in applicable if c.status == ImplementationStatus.IMPLEMENTED
        )
        return round(100 * implemented / len(applicable), 1)

    def to_csv(self, path: Path) -> None:
        """Export tracker state to CSV for audit trail handoff."""
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "subcategory_code", "description", "status",
                    "evidence_reference", "owner", "target_date", "notes"
                ],
            )
            writer.writeheader()
            for ctrl in self.controls.values():
                writer.writerow(asdict(ctrl))


# ---------------------------------------------------------------------------
# 6. Dual-Framework Mapping Report Generator
# ---------------------------------------------------------------------------

EU_NIST_CROSSWALK: List[Dict[str, str]] = [
    {
        "eu_article": "Article 9 — Risk Management",
        "annex_iv_section": "§4 Validation & Testing",
        "nist_function": "MANAGE",
        "nist_subcategory": "MS-1.1",
        "nist_description": "AI system risks are identified and assessed",
        "compliance_notes": "Risk register + test evidence required for both frameworks",
    },
    {
        "eu_article": "Article 10 — Data Governance",
        "annex_iv_section": "§3 Training Data",
        "nist_function": "MAP",
        "nist_subcategory": "MP-2.1",
        "nist_description": "Scientific findings used in AI design are identified",
        "compliance_notes": "Data cards and provenance documentation satisfy both",
    },
    {
        "eu_article": "Article 11 — Technical Documentation",
        "annex_iv_section": "Full Annex IV",
        "nist_function": "GOVERN",
        "nist_subcategory": "GV-1.1",
        "nist_description": "Policies and practices for TEVV",
        "compliance_notes": "Annex IV YAML package maps directly to GV-1.1 evidence",
    },
    {
        "eu_article": "Article 12 — Record Keeping",
        "annex_iv_section": "§6 Post-Market",
        "nist_function": "MANAGE",
        "nist_subcategory": "MG-4.1",
        "nist_description": "Post-deployment risks and benefits are monitored",
        "compliance_notes": "TamperEvidentAuditLog satisfies both obligations",
    },
    {
        "eu_article": "Article 13 — Transparency",
        "annex_iv_section": "§1 General Description",
        "nist_function": "GOVERN",
        "nist_subcategory": "GV-1.2",
        "nist_description": "Organizational commitment to AI risk governance",
        "compliance_notes": "User-facing disclosure docs + internal governance charter",
    },
    {
        "eu_article": "Article 14 — Human Oversight",
        "annex_iv_section": "§4 Human Oversight Measures",
        "nist_function": "MANAGE",
        "nist_subcategory": "MG-2.2",
        "nist_description": "Mechanisms to assess and adjust",
        "compliance_notes": "Override capability and escalation paths documented",
    },
    {
        "eu_article": "Article 15 — Accuracy / Robustness",
        "annex_iv_section": "§5 Technical Standards",
        "nist_function": "MEASURE",
        "nist_subcategory": "MS-2.5",
        "nist_description": "Rigorous testing before high-risk deployment",
        "compliance_notes": "Benchmark suite + adversarial evaluation report",
    },
    {
        "eu_article": "Article 72 — Post-Market Monitoring",
        "annex_iv_section": "§6 Post-Market Plan",
        "nist_function": "MANAGE",
        "nist_subcategory": "MG-4.2",
        "nist_description": "Evaluations used to make adjustments",
        "compliance_notes": "PostMarketMonitoringReport dataclass + monthly cadence",
    },
]


def generate_dual_framework_report(
    package: AnnexIVPackage,
    tracker: NISTAI6001Tracker,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate a combined EU AI Act + NIST AI RMF compliance report.

    Parameters
    ----------
    package : AnnexIVPackage
        The Annex IV documentation bundle.
    tracker : NISTAI6001Tracker
        The NIST control tracker for the same system.
    output_path : Path, optional
        If provided, write the report as JSON to this path.

    Returns
    -------
    dict with keys: metadata, annex_iv_status, nist_status, crosswalk, recommendations
    """
    nist_gap = tracker.gap_report()
    annex_score = package.completeness_score()
    missing = package.missing_required_fields()

    recommendations: List[str] = []
    if missing:
        recommendations.append(
            f"Complete {len(missing)} missing Annex IV fields before deployment: {missing}"
        )
    for gap_ctrl in nist_gap["gaps"]:
        recommendations.append(
            f"NIST {gap_ctrl['subcategory_code']} ({gap_ctrl['status']}): "
            f"{gap_ctrl['description']}"
        )

    report = {
        "metadata": {
            "system_name": package.system_name,
            "system_version": package.system_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_version": "1.0",
        },
        "annex_iv_status": {
            "completeness_score": annex_score,
            "passes_ci_gate": annex_score >= ANNEX_IV_CI_THRESHOLD,
            "missing_required_fields": missing,
        },
        "nist_status": {
            "completion_percentage": tracker.completion_percentage(),
            "summary": nist_gap["summary"],
            "gap_count": len(nist_gap["gaps"]),
        },
        "crosswalk": EU_NIST_CROSSWALK,
        "recommendations": recommendations,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"[DualFramework] Report written to {output_path}")

    return report


# ---------------------------------------------------------------------------
# 7. PostMarketMonitoringReport
# ---------------------------------------------------------------------------

@dataclass
class MetricSnapshot:
    """A single metric measurement at a point in time."""
    metric_name: str
    value: float
    unit: str
    threshold: float
    breached: bool = field(init=False)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        self.breached = self.value > self.threshold


@dataclass
class PostMarketMonitoringReport:
    """
    Article 72 EU AI Act — Post-Market Monitoring report.

    Covers performance drift, incident counts, user complaints, bias signals,
    and remediation actions taken since the last reporting period.
    """
    system_name: str
    system_version: str
    reporting_period_start: str
    reporting_period_end: str
    report_author: str

    # Core monitoring data
    total_requests: int = 0
    flagged_outputs: int = 0
    user_complaints: int = 0
    serious_incidents: int = 0       # EU AI Act Article 73 threshold
    near_miss_incidents: int = 0
    metric_snapshots: List[MetricSnapshot] = field(default_factory=list)

    # Drift and bias
    accuracy_drift_pct: float = 0.0  # positive = degradation
    bias_signal_triggered: bool = False
    bias_signal_details: str = ""

    # Remediation
    model_updated: bool = False
    guardrails_updated: bool = False
    remediation_actions: List[str] = field(default_factory=list)

    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    next_report_due: str = ""

    def flagged_rate(self) -> float:
        """Fraction of requests that produced flagged outputs."""
        if self.total_requests == 0:
            return 0.0
        return round(self.flagged_outputs / self.total_requests, 6)

    def requires_notified_body_report(self) -> bool:
        """
        EU AI Act Article 73: serious incidents affecting health/safety/rights
        require notification to the market surveillance authority.
        Threshold: any serious_incidents > 0.
        """
        return self.serious_incidents > 0

    def add_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        threshold: float,
    ) -> MetricSnapshot:
        snap = MetricSnapshot(metric_name=metric_name, value=value, unit=unit, threshold=threshold)
        self.metric_snapshots.append(snap)
        return snap

    def breached_metrics(self) -> List[MetricSnapshot]:
        return [m for m in self.metric_snapshots if m.breached]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["flagged_rate"] = self.flagged_rate()
        d["requires_notified_body_report"] = self.requires_notified_body_report()
        d["breached_metric_count"] = len(self.breached_metrics())
        return d

    def to_json(self, path: Optional[Path] = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
        return payload


# ---------------------------------------------------------------------------
# __main__ — demonstrate all components end-to-end
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("Chapter 11 — EU AI Act + NIST AI RMF Compliance Demo")
    print("=" * 70)

    # --- 1. Build a partially complete Annex IV package ---
    pkg = AnnexIVPackage(
        system_name="CustomerCareBot",
        system_version="2.1.0",
        intended_purpose="Automated tier-1 customer support for retail banking",
        risk_category="high",
        provider_name="Acme Financial AI Ltd.",
        provider_address="123 Innovation Drive, Dublin, Ireland",
        contact_email="ai-compliance@acme.example",
        general_description=(
            "GPT-4-based chat assistant handling account queries, "
            "dispute initiation, and product information for retail customers."
        ),
        design_specifications="See design-spec-v2.1.pdf in the compliance vault.",
        training_data_summary="Fine-tuned on 2.3M anonymized support transcripts (2021-2023).",
        validation_testing="See evaluation-report-v2.1.pdf; accuracy 91.4% on holdout set.",
        technical_standards="ISO/IEC 42001:2023, EN 301 549",
        post_market_plan="Monthly drift monitoring; quarterly bias audit; annual red-team exercise.",
        human_oversight_measures="Agents can escalate any conversation; override available at all times.",
        accuracy_metrics="F1=0.914 on intent classification; BLEU=0.71 on generative responses.",
        robustness_measures="Adversarial prompt testing quarterly; input length limits enforced.",
        cybersecurity_measures="OWASP LLM Top 10 mitigations applied; rate limiting; PII scrubbing.",
        declaration_of_conformity="DoC-CCBOT-2024-001 signed by Chief Compliance Officer.",
    )
    print(f"\nAnnex IV completeness: {pkg.completeness_score():.2%}")
    print(f"Missing required fields: {pkg.missing_required_fields()}")

    # --- 2. Run CI gate ---
    print("\n--- Annex IV CI Gate ---")
    run_annex_iv_ci_gate(pkg, strict=False)

    # --- 3. Provenance recorder ---
    print("\n--- Output Provenance Recorder ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = OutputProvenanceRecorder(Path(tmpdir) / "provenance.jsonl")
        rec = recorder.record(
            prompt="What is my account balance?",
            output="Your current balance is €1,240.00.",
            model_id="gpt-4o",
            model_version="2024-08-06",
            temperature=0.0,
            max_tokens=256,
            system_prompt_version="v3.1",
            user_id="u-42",
            session_id="sess-abc123",
        )
        print(f"Provenance record ID: {rec.record_id}")
        print(f"Signature valid: {rec.verify()}")

        # Demonstrate tamper detection
        rec.output_hash = "tampered_hash"
        print(f"Signature valid after tamper: {rec.verify()}")

    # --- 4. TamperEvidentAuditLog ---
    print("\n--- Tamper-Evident Audit Log ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        log = TamperEvidentAuditLog(Path(tmpdir) / "audit.jsonl")
        log.append("model_deployment", {"version": "2.1.0", "deployed_by": "mlops-pipeline"})
        log.append("policy_update", {"policy": "rate_limit", "new_value": 100})
        log.append("incident_detected", {"severity": "low", "description": "Unusual prompt pattern"})
        print(f"Entries written: {log._seq}")
        log.verify_integrity()

    # --- 5. NIST AI 600-1 Tracker ---
    print("\n--- NIST AI 600-1 Tracker ---")
    tracker = NISTAI6001Tracker("CustomerCareBot")
    tracker.update("GV-1.1", ImplementationStatus.IMPLEMENTED,
                   evidence_reference="governance-policy-v3.pdf", owner="AI Governance Team")
    tracker.update("GV-1.2", ImplementationStatus.IMPLEMENTED,
                   evidence_reference="board-charter.pdf", owner="CTO Office")
    tracker.update("MS-1.1", ImplementationStatus.IN_PROGRESS,
                   owner="Risk Team", target_date="2024-09-30")
    tracker.update("MS-2.6", ImplementationStatus.NOT_STARTED,
                   owner="ML Engineering", target_date="2024-10-31")
    tracker.update("MG-4.1", ImplementationStatus.IMPLEMENTED,
                   evidence_reference="monitoring-runbook-v2.pdf", owner="MLOps Team")
    gap = tracker.gap_report()
    print(f"Completion: {tracker.completion_percentage()}%")
    print(f"Summary: {gap['summary']}")
    print(f"Gap count: {len(gap['gaps'])}")

    # --- 6. Dual-framework mapping report ---
    print("\n--- Dual-Framework Mapping Report ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        report = generate_dual_framework_report(
            pkg, tracker, output_path=Path(tmpdir) / "compliance-report.json"
        )
        print(f"Annex IV passes CI gate: {report['annex_iv_status']['passes_ci_gate']}")
        print(f"NIST completion: {report['nist_status']['completion_percentage']}%")
        print(f"Recommendations: {len(report['recommendations'])}")

    # --- 7. PostMarketMonitoringReport ---
    print("\n--- Post-Market Monitoring Report ---")
    pmm = PostMarketMonitoringReport(
        system_name="CustomerCareBot",
        system_version="2.1.0",
        reporting_period_start="2024-07-01",
        reporting_period_end="2024-07-31",
        report_author="AI Risk Lead",
        total_requests=148_320,
        flagged_outputs=412,
        user_complaints=17,
        serious_incidents=0,
        near_miss_incidents=3,
        accuracy_drift_pct=0.8,
        bias_signal_triggered=False,
        remediation_actions=["Prompt injection filter threshold tightened on 2024-07-14"],
        next_report_due="2024-08-31",
    )
    pmm.add_metric("hallucination_rate", 0.031, "fraction", threshold=0.05)
    pmm.add_metric("avg_latency_p99_ms", 1840.0, "ms", threshold=2000.0)
    pmm.add_metric("refusal_rate", 0.0027, "fraction", threshold=0.01)
    print(f"Flagged rate: {pmm.flagged_rate():.4%}")
    print(f"Breached metrics: {[m.metric_name for m in pmm.breached_metrics()]}")
    print(f"Requires notified body report: {pmm.requires_notified_body_report()}")

    print("\n" + "=" * 70)
    print("All Chapter 11 components demonstrated successfully.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# === Sections 10.7-10.8: Post-Market Monitoring and AI Incident Response ===
# ---------------------------------------------------------------------------
# These classes implement the serious-incident escalation pipeline (10.7),
# the EU AI Act Article 73 notification package (10.7.3), the automated
# incident state capture tool (10.8.4), the containment runbook (10.8.1),
# and the incident classifier (10.8.2).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 10.7.1 IncidentEscalation — Article 73 SLA-enforced escalation pipeline
# Listing 10.8a
# ---------------------------------------------------------------------------

class EscalationPhase(str, Enum):
    DETECTED = "detected"               # monitoring system fired alert
    ACKNOWLEDGED = "acknowledged"       # on-call engineer acknowledged
    CLASSIFIED = "classified"           # incident classified as potential serious / false positive
    LEGAL_REVIEWED = "legal_reviewed"   # legal team reviewed classification
    NOTIFIED = "notified"               # Article 73 notification submitted
    CLOSED = "closed"                   # incident resolved and closed


# SLA hours for each phase transition (measured from system detection time)
PHASE_SLA_HOURS: Dict[EscalationPhase, int] = {
    EscalationPhase.ACKNOWLEDGED: 24,
    EscalationPhase.CLASSIFIED: 48,
    EscalationPhase.LEGAL_REVIEWED: 120,   # 5 days from detection
    EscalationPhase.NOTIFIED: 360,         # 15 days from detection (Article 73)
}


@dataclass
class PhaseRecord:
    phase: EscalationPhase
    completed_at: Optional[datetime] = None
    completed_by: str = ""
    notes: str = ""


@dataclass
class IncidentEscalation:
    """
    Models a serious incident escalation pipeline with Article 73 SLA enforcement.

    The detection_time is set at instantiation and never changes: it marks
    the moment the monitoring system fired the alert.  Every SLA deadline is
    computed relative to that timestamp.
    """
    incident_id: str
    agent_name: str
    detection_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    current_phase: EscalationPhase = EscalationPhase.DETECTED
    phase_records: List[PhaseRecord] = field(default_factory=list)
    is_article_73_candidate: bool = False   # set True during CLASSIFIED phase

    def advance(
        self,
        phase: EscalationPhase,
        completed_by: str,
        notes: str = "",
        is_article_73_candidate: Optional[bool] = None,
    ) -> None:
        """Record completion of a phase.  Call once per phase transition."""
        record = PhaseRecord(
            phase=phase,
            completed_at=datetime.now(timezone.utc),
            completed_by=completed_by,
            notes=notes,
        )
        self.phase_records.append(record)
        self.current_phase = phase
        if is_article_73_candidate is not None:
            self.is_article_73_candidate = is_article_73_candidate

    def _deadline(self, phase: EscalationPhase) -> datetime:
        from datetime import timedelta
        hours = PHASE_SLA_HOURS[phase]
        return self.detection_time + timedelta(hours=hours)

    def sla_status(self) -> Dict[str, Any]:
        """
        Returns the SLA state for every phase that has a deadline.
        Surface any 'breached' or 'warning' entries to the on-call engineer
        every six hours during an active serious incident.
        """
        now = datetime.now(timezone.utc)
        phases_done = {r.phase for r in self.phase_records}
        status: Dict[str, Any] = {}

        for phase, hours in PHASE_SLA_HOURS.items():
            deadline = self._deadline(phase)
            remaining_hours = (deadline - now).total_seconds() / 3600

            if phase in phases_done:
                record = next(r for r in self.phase_records if r.phase == phase)
                state = "complete"
                completed_at = record.completed_at.isoformat() if record.completed_at else None
            elif remaining_hours < 0:
                state = "breached"
                completed_at = None
            elif remaining_hours < 24:
                state = "warning"
                completed_at = None
            else:
                state = "ok"
                completed_at = None

            status[phase.value] = {
                "state": state,
                "deadline": deadline.isoformat(),
                "remaining_hours": round(remaining_hours, 1),
                "completed_at": completed_at,
            }

        # Always show the Article 73 absolute deadline prominently
        status["article_73_deadline"] = self._deadline(
            EscalationPhase.NOTIFIED
        ).isoformat()
        status["is_article_73_candidate"] = self.is_article_73_candidate
        return status

    def to_json(self, output_path: Optional[str] = None) -> str:
        """Serialize the full escalation record.  Pass output_path to write to disk."""
        data = {
            "incident_id": self.incident_id,
            "agent_name": self.agent_name,
            "detection_time": self.detection_time.isoformat(),
            "current_phase": self.current_phase.value,
            "is_article_73_candidate": self.is_article_73_candidate,
            "article_73_deadline": self._deadline(
                EscalationPhase.NOTIFIED
            ).isoformat(),
            "phase_records": [
                {
                    "phase": r.phase.value,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    "completed_by": r.completed_by,
                    "notes": r.notes,
                }
                for r in self.phase_records
            ],
            "sla_status": self.sla_status(),
        }
        result = json.dumps(data, indent=2)
        if output_path:
            Path(output_path).write_text(result)
        return result


# ---------------------------------------------------------------------------
# 10.7.3 Article73NotificationPackage — 72h and 15-day notification packages
# Listing 10.8b
# ---------------------------------------------------------------------------

@dataclass
class Article73NotificationPackage:
    """
    Tracks the completeness of the 72-hour and 15-day EU AI Act Article 73
    notification packages and serializes them for internal legal review.
    """
    incident_id: str
    awareness_timestamp: float       # unix timestamp: when you became aware
    incident_timestamp: float        # unix timestamp: when the incident occurred
    agent_name: str
    system_description: str
    affected_users_count: int        # 0 if not yet determined
    incident_description: str
    initial_impact_assessment: str
    containment_actions: List[str] = field(default_factory=list)
    root_cause: str = ""
    corrective_measures: List[str] = field(default_factory=list)
    monitoring_improvements: List[str] = field(default_factory=list)

    def is_72h_complete(self) -> bool:
        """Returns True when all fields required for the 72-hour package are populated."""
        required = ["incident_description", "initial_impact_assessment", "containment_actions"]
        return all(getattr(self, f) for f in required)

    def is_15d_complete(self) -> bool:
        """Returns True when all fields required for the 15-day package are populated."""
        required_15d = ["root_cause", "corrective_measures", "monitoring_improvements"]
        return self.is_72h_complete() and all(getattr(self, f) for f in required_15d)

    def days_until_deadline(self) -> float:
        """Returns days remaining until the 15-day notification deadline (0.0 when past)."""
        elapsed = (time.time() - self.awareness_timestamp) / 86400
        return max(0.0, 15.0 - elapsed)

    def to_json(self, output_path: Optional[str] = None) -> str:
        """Serialize the notification package.  Pass output_path to write to disk."""
        data = {
            "incident_id": self.incident_id,
            "agent_name": self.agent_name,
            "awareness_timestamp": self.awareness_timestamp,
            "awareness_datetime_utc": datetime.fromtimestamp(
                self.awareness_timestamp, tz=timezone.utc
            ).isoformat(),
            "incident_timestamp": self.incident_timestamp,
            "incident_datetime_utc": datetime.fromtimestamp(
                self.incident_timestamp, tz=timezone.utc
            ).isoformat(),
            "system_description": self.system_description,
            "affected_users_count": self.affected_users_count,
            "72h_package": {
                "complete": self.is_72h_complete(),
                "incident_description": self.incident_description,
                "initial_impact_assessment": self.initial_impact_assessment,
                "containment_actions": self.containment_actions,
            },
            "15d_package": {
                "complete": self.is_15d_complete(),
                "root_cause": self.root_cause,
                "corrective_measures": self.corrective_measures,
                "monitoring_improvements": self.monitoring_improvements,
            },
            "days_until_deadline": self.days_until_deadline(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        result = json.dumps(data, indent=2)
        if output_path:
            Path(output_path).write_text(result)
        return result


# ---------------------------------------------------------------------------
# 10.8.1 ContainmentRunbook — four pre-planned containment steps
# Listing 10.7a
# ---------------------------------------------------------------------------

@dataclass
class ContainmentResult:
    """Records the outcome of executing the containment runbook."""
    agent_id: str
    incident_type: str
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    completed_at: float = field(default_factory=time.time)
    fully_contained: bool = False


class ContainmentRunbook:
    """
    Executes the four pre-planned containment steps for an agent incident.

    Wire execute_containment() directly to your tripwire event handler and
    CUSUM escalation path.  Each step is independent: a failure in one step
    does not prevent the remaining steps from executing.
    """

    def __init__(self, fallback_router, memory_store, credential_manager, state_capture):
        self.router = fallback_router
        self.memory = memory_store
        self.credentials = credential_manager
        self.capture = state_capture

    def execute_containment(self, agent_id: str, incident_type: str) -> ContainmentResult:
        """
        Execute all four containment steps and return a ContainmentResult.

        Steps
        -----
        1. Route new requests to fallback handler (stops new work reaching the agent).
        2. Freeze agent memory (writes to frozen state fail safely).
        3. Revoke API credentials (no further external actions even if process runs).
        4. Snapshot agent state (planning logs, action history, memory, tripwire event).

        The ContainmentResult.fully_contained flag is True only when all four
        steps complete without error.
        """
        result = ContainmentResult(agent_id=agent_id, incident_type=incident_type)

        # Step 1: Route new requests to fallback handler
        try:
            self.router.enable_fallback(agent_id)
            result.steps_completed.append("fallback_routing_enabled")
        except Exception as e:
            result.steps_failed.append(f"fallback_routing: {e}")

        # Step 2: Freeze agent memory
        try:
            self.memory.freeze(agent_id)
            result.steps_completed.append("memory_frozen")
        except Exception as e:
            result.steps_failed.append(f"memory_freeze: {e}")

        # Step 3: Revoke API credentials
        try:
            self.credentials.revoke(agent_id)
            result.steps_completed.append("credentials_revoked")
        except Exception as e:
            result.steps_failed.append(f"credential_revocation: {e}")

        # Step 4: Snapshot agent state for post-mortem
        try:
            self.capture.snapshot(agent_id, incident_type=incident_type)
            result.steps_completed.append("state_snapshot_captured")
        except Exception as e:
            result.steps_failed.append(f"state_snapshot: {e}")

        result.fully_contained = len(result.steps_failed) == 0
        return result


# ---------------------------------------------------------------------------
# 10.8.4 IncidentCapture — automated P0/P1 state capture
# Listing 10.7b
# ---------------------------------------------------------------------------

class IncidentCapture:
    """
    Captures complete agent state when a P0/P1 tripwire fires.

    Uses only the Python standard library to avoid import failures during an
    active incident.  Packages all required artifacts into a single JSON file
    keyed by a SHA-256-derived incident ID.
    """

    def __init__(self, incident_dir: str = "/var/log/ai-incidents"):
        self.incident_dir = Path(incident_dir)
        self.incident_dir.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        session_id: str,
        severity: str,
        tripwire_event: Dict[str, Any],
        action_log: List[Dict[str, Any]],
        memory_segments: List[Dict[str, Any]],
        agent_name: str,
    ) -> str:
        """
        Package agent state into a JSON artifact and write it to incident_dir.

        Parameters
        ----------
        session_id : str
            The agent session identifier from your telemetry layer (Langfuse, etc.).
        severity : str
            "P0" or "P1".
        tripwire_event : dict
            The raw TripwireEvent dict that fired the alert (rule_name, context, timestamp).
        action_log : list[dict]
            The agent's full action history for the session.  Truncated to last 50.
        memory_segments : list[dict]
            Current memory snapshot.  Truncated to last 20 segments.
        agent_name : str
            Logical name of the agent (matches the Annex IV system_name).

        Returns
        -------
        str
            The incident_id assigned to this capture (12-character hex prefix).
        """
        incident_id = hashlib.sha256(
            f"{session_id}{time.time()}".encode()
        ).hexdigest()[:12]

        package = {
            "incident_id": incident_id,
            "captured_at": time.time(),
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "agent_name": agent_name,
            "severity": severity,
            "tripwire": tripwire_event,
            "action_log_tail": action_log[-50:],          # last 50 actions for triage
            "memory_snapshot": memory_segments[-20:],      # last 20 segments
            "action_log_total_count": len(action_log),
            "memory_segment_total_count": len(memory_segments),
        }

        output_path = self.incident_dir / f"incident-{incident_id}.json"
        output_path.write_text(json.dumps(package, indent=2))
        return incident_id


# ---------------------------------------------------------------------------
# 10.8.2 IncidentClassifier — P0/P1/P2 severity classification
# Derived from section 10.8.2 severity criteria and chapter 7 telemetry fields
# ---------------------------------------------------------------------------

@dataclass
class ClassificationInput:
    """
    Telemetry fields from chapter 7's monitoring layer used to classify severity.

    Fields
    ------
    tripwire_fired : bool
        True if any tripwire rule fired during the session.
    tripwire_rule : str
        The rule name that fired (e.g., "WRITE_WITHOUT_READ", "UNAUTHORIZED_TOOL").
        Empty string if no tripwire fired.
    action_was_irreversible : bool
        True if the agent's action is confirmed to have had an irreversible external
        effect (email sent, production DB write, payment initiated, file deleted).
    cusum_alert : bool
        True if the CUSUM action-rate monitor crossed the detection threshold (h).
    cognitive_degradation_level : int
        0 = none; 1 = minor (single-step deviation); 2 = moderate (continuing after
        tool error without plan adjustment); 3 = severe (goal substitution detected).
    scope_violation_confirmed : bool
        True if the agent accessed a resource outside its authorized tool allowlist,
        regardless of whether the access succeeded.
    """
    tripwire_fired: bool = False
    tripwire_rule: str = ""
    action_was_irreversible: bool = False
    cusum_alert: bool = False
    cognitive_degradation_level: int = 0
    scope_violation_confirmed: bool = False


@dataclass
class ClassificationResult:
    """
    Output of IncidentClassifier.classify().

    Fields
    ------
    severity : str
        "P0", "P1", or "P2".
    rationale : str
        Human-readable explanation of why this severity was assigned.
    article_73_clock_starts : bool
        True for P0 incidents.  The 15-day Article 73 notification clock starts
        at the moment you become aware of a P0 incident.
    resolve_within_hours : int
        SLA target: 1 hour for P0, 4 hours for P1, 24 hours for P2.
    """
    severity: str
    rationale: str
    article_73_clock_starts: bool
    resolve_within_hours: int


class IncidentClassifier:
    """
    Classifies AI agent incidents as P0, P1, or P2 using the severity criteria
    from section 10.8.2 and the telemetry fields produced by chapter 7's
    monitoring layer.

    Classification logic
    --------------------
    P0 (Critical, 1h SLA): tripwire fired AND action confirmed irreversible.
        The agent took an action outside its authorized scope that cannot be
        undone (email sent, production DB write, payment, file deletion).
        The Article 73 clock starts here.

    P1 (High, 4h SLA): tripwire fired OR scope violation confirmed, but
        irreversibility is NOT confirmed.  Impact may still be reversible.
        Containment required before downgrade to P2 is permitted.

    P2 (Medium, 24h SLA): anomalous telemetry without confirmed scope violation
        or irreversible action.  CUSUM alert alone, or cognitive degradation
        Level 1–2 without an accompanying tripwire, maps here.

    No incident: no tripwire, no CUSUM alert, no degradation signal.
    """

    def classify(self, inputs: ClassificationInput) -> ClassificationResult:
        """
        Classify an incident and return a ClassificationResult.

        Parameters
        ----------
        inputs : ClassificationInput
            Telemetry snapshot at time of incident detection.

        Returns
        -------
        ClassificationResult
        """
        # P0: irreversible external action confirmed
        if inputs.tripwire_fired and inputs.action_was_irreversible:
            return ClassificationResult(
                severity="P0",
                rationale=(
                    f"Tripwire '{inputs.tripwire_rule}' fired and the action is confirmed "
                    f"irreversible. Immediate containment required. Article 73 clock starts."
                ),
                article_73_clock_starts=True,
                resolve_within_hours=1,
            )

        # P1: scope violation or tripwire without confirmed irreversibility
        if inputs.tripwire_fired or inputs.scope_violation_confirmed:
            return ClassificationResult(
                severity="P1",
                rationale=(
                    f"{'Tripwire fired' if inputs.tripwire_fired else 'Scope violation detected'} "
                    f"but irreversibility not yet confirmed. Blast radius assessment required "
                    f"before downgrading to P2."
                ),
                article_73_clock_starts=False,
                resolve_within_hours=4,
            )

        # P1 escalation path: severe cognitive degradation with CUSUM
        if inputs.cusum_alert and inputs.cognitive_degradation_level >= 3:
            return ClassificationResult(
                severity="P1",
                rationale=(
                    f"CUSUM alert combined with cognitive degradation Level "
                    f"{inputs.cognitive_degradation_level} (goal substitution). "
                    f"Escalated from P2 due to severity of degradation signal."
                ),
                article_73_clock_starts=False,
                resolve_within_hours=4,
            )

        # P2: anomalous telemetry, no confirmed scope violation
        if inputs.cusum_alert or inputs.cognitive_degradation_level >= 1:
            return ClassificationResult(
                severity="P2",
                rationale=(
                    f"Anomalous telemetry detected "
                    f"({'CUSUM alert' if inputs.cusum_alert else ''}"
                    f"{', ' if inputs.cusum_alert and inputs.cognitive_degradation_level >= 1 else ''}"
                    f"{'cognitive degradation Level ' + str(inputs.cognitive_degradation_level) if inputs.cognitive_degradation_level >= 1 else ''}"
                    f") without confirmed scope violation or irreversible action. "
                    f"Investigation required; containment not mandatory."
                ),
                article_73_clock_starts=False,
                resolve_within_hours=24,
            )

        # No incident
        return ClassificationResult(
            severity="CLEAR",
            rationale="No tripwire, CUSUM alert, or cognitive degradation signal detected.",
            article_73_clock_starts=False,
            resolve_within_hours=0,
        )
