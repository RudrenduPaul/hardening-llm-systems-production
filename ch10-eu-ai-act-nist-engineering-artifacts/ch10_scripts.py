"""
Chapter 10 — EU AI Act + NIST AI RMF Compliance Automation
===========================================================
Manning book: "Hardening LLM Systems in Production" by Rudrendu Paul

Companion script covering:
  - Annex IV artifact index generator (section 10.3, Listing 10.0)
  - AnnexIVPackage dataclass with completeness_score() (Listing 10.1)
  - Annex IV CI gate (sys.exit(1) on failure) (Listing 10.2)
  - Output provenance recorder with HMAC-SHA256 signature (Listing 10.3)
  - TamperEvidentAuditLog with chained-hash tamper detection (Listing 10.4)
  - NIST AI 600-1 triage report generator (Listing 10.4b)
  - NISTAI6001Tracker with ImplementationStatus enum and gap_report() (Listing 10.5)
  - Dual-framework mapping report (EU AI Act + NIST cross-reference) (Listing 10.6)
  - IncidentEscalation — Article 73 SLA escalation pipeline (Listing 10.7)
  - Article73NotificationPackage — 72h/15-day notification packages (Listing 10.8)
  - PostMarketMonitoringReport dataclass (Listing 10.9)
  - ContainmentRunbook — four pre-planned containment steps (Listing 10.10)
  - IncidentCapture — automated P0/P1 state capture (Listing 10.11)
  - CI/CD Annex IV completeness gate with version + freshness checks (Listing 10.12)
  - IncidentClassifier — P0/P1/P2 severity classification (sections 10.7-10.8,
    described in prose only; not yet assigned a listing number, see README)

Dependencies: pyyaml>=6.0  (all others are stdlib)
"""

from __future__ import annotations

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
# 1. Annex IV artifact index generator (section 10.3)
# Listing 10.0
# ---------------------------------------------------------------------------

ANNEX_IV_ARTIFACT_MAP: Dict[str, Dict[str, Any]] = {
    "section_1_general_description": {
        "required": ["architecture_diagram_path", "model_card_path", "intended_use_policy_path"],
        "owner": "Platform engineering + ML engineering",
        "cadence": "On major version / model update / use case change",
        "format": "PDF + versioned Markdown + PDF",
    },
    "section_2_system_elements": {
        "required": ["data_pipeline_diagram_path", "training_data_doc_path",
                     "model_version_history_path", "finetuning_records_path"],
        "owner": "ML engineering",
        "cadence": "On pipeline change / dataset change / model update",
        "format": "PDF + Markdown CHANGELOG",
    },
    "section_3_monitoring_control": {
        "required": ["dashboard_config_path", "alert_runbook_path",
                     "human_oversight_protocol_path", "operational_procedures_path"],
        "owner": "Platform engineering + on-call rotation",
        "cadence": "On threshold change / on-call rotation change",
        "format": "JSON export + PDF (named owner)",
    },
    "section_4_risk_management": {
        "required": ["risk_assessment_path", "red_team_report_path",
                     "bias_assessment_path", "adversarial_robustness_path"],
        "owner": "Security + ML fairness",
        "cadence": "Quarterly + after major changes",
        "format": "PDF + JSON",
    },
    "section_5_lifecycle_changes": {
        "required": ["changelog_path", "model_update_log_path",
                     "prompt_change_log_path", "deployment_approval_path"],
        "owner": "Engineering + ML/product engineering + engineering management",
        "cadence": "On every relevant change event",
        "format": "CHANGELOG.md + signed PDF",
    },
    "section_6_conformity_assessment": {
        "required": ["evaluation_results_path", "third_party_audit_path", "self_assessment_path"],
        "owner": "ML engineering + engineering leadership",
        "cadence": "Every model version / before each production deploy",
        "format": "JSON/CSV + signed PDF",
    },
}


def generate_annex_iv_index(
    deployment_config: Dict[str, str],
    max_artifact_age_days: int = 30,
) -> Dict[str, Any]:
    """
    Checks a deployment's artifact paths against ANNEX_IV_ARTIFACT_MAP.

    Parameters
    ----------
    deployment_config : dict
        Maps each required artifact key (e.g. "architecture_diagram_path")
        to its filesystem path.
    max_artifact_age_days : int
        Artifacts older than this are flagged as stale.

    Returns
    -------
    dict with keys: sections, sections_with_gaps, sections_with_stale_artifacts,
    ready_for_audit
    """
    now = time.time()
    sections: Dict[str, Any] = {}
    sections_with_gaps: List[str] = []
    sections_with_stale_artifacts: List[str] = []

    for section, spec in ANNEX_IV_ARTIFACT_MAP.items():
        present, missing, stale = [], [], []
        for key in spec["required"]:
            path = deployment_config.get(key)
            if not path or not os.path.exists(path):
                missing.append(key)
                continue
            present.append(key)
            age_days = (now - os.path.getmtime(path)) / 86400
            if age_days > max_artifact_age_days:
                stale.append(key)

        sections[section] = {
            "owner": spec["owner"],
            "cadence": spec["cadence"],
            "format": spec["format"],
            "present": present,
            "missing": missing,
            "stale": stale,
        }
        if missing:
            sections_with_gaps.append(section)
        if stale:
            sections_with_stale_artifacts.append(section)

    return {
        "generated_at": now,
        "sections": sections,
        "sections_with_gaps": sections_with_gaps,
        "sections_with_stale_artifacts": sections_with_stale_artifacts,
        "ready_for_audit": not sections_with_gaps and not sections_with_stale_artifacts,
    }


# ---------------------------------------------------------------------------
# 2. AnnexIVPackage — EU AI Act Article 11 / Annex IV documentation bundle
# Listing 10.1
# ---------------------------------------------------------------------------

ANNEX_IV_REQUIRED_FIELDS: List[str] = [
    "system_name", "system_version", "intended_purpose",
    "model_family", "model_version", "training_data_description",
    "architecture_description", "human_oversight_design",
    "risk_categories_addressed",
]

# Fields whose value is a filesystem path; completeness also checks these
# actually exist on disk, not just that the field is non-empty.
ANNEX_IV_ARTIFACT_PATH_FIELDS: List[str] = [
    "red_team_report_path", "bias_assessment_path",
    "evaluation_results_path", "adversarial_robustness_path",
]


@dataclass
class AnnexIVPackage:
    """Assembles an EU AI Act Annex IV documentation package from CI/CD artifacts."""

    # General description (Annex IV, point 1)
    system_name: str
    system_version: str
    intended_purpose: str
    deployment_date: str
    operator_name: str

    # System elements (Annex IV, point 2)
    model_family: str
    model_version: str
    training_data_description: str
    architecture_description: str
    components: List[str] = field(default_factory=list)

    # Monitoring and control (Annex IV, point 3)
    monitoring_metrics: List[str] = field(default_factory=list)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    human_oversight_design: str = ""

    # Risk management (Annex IV, point 4)
    risk_categories_addressed: List[str] = field(default_factory=list)
    red_team_report_path: str = ""
    bias_assessment_path: str = ""
    adversarial_robustness_path: str = ""

    # Conformity assessment (Annex IV, point 6)
    evaluation_results_path: str = ""
    self_assessment_path: str = ""

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "annex-iv-v2.0"

    def missing_required_fields(self) -> List[str]:
        """Return the required fields (section 10.3.3) that are empty."""
        return [f for f in ANNEX_IV_REQUIRED_FIELDS if not getattr(self, f, None)]

    def missing_or_stale_artifacts(self) -> List[str]:
        """Artifact-path fields that are unset or point to a file that doesn't exist."""
        return [
            f for f in ANNEX_IV_ARTIFACT_PATH_FIELDS
            if not getattr(self, f, "") or not Path(getattr(self, f)).exists()
        ]

    def completeness_score(self) -> float:
        """
        Checks two distinct things: whether the required fields are populated
        with non-empty values (75% weight), and whether the referenced
        artifact files actually exist on disk (25% weight). A package with
        valid-looking fields pointing at files that were never generated
        scores below 1.0 here even though every field is "filled in".
        """
        req_filled = len(ANNEX_IV_REQUIRED_FIELDS) - len(self.missing_required_fields())
        req_score = (req_filled / len(ANNEX_IV_REQUIRED_FIELDS)) * 0.75

        artifacts_ok = len(ANNEX_IV_ARTIFACT_PATH_FIELDS) - len(self.missing_or_stale_artifacts())
        artifact_score = (artifacts_ok / len(ANNEX_IV_ARTIFACT_PATH_FIELDS)) * 0.25

        return round(req_score + artifact_score, 4)

    def to_yaml(self) -> str:
        """Serialize package to a YAML string for storage / version control."""
        return yaml.dump(asdict(self), default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "AnnexIVPackage":
        """Deserialize from a YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, output_path: Optional[Path] = None) -> str:
        """Serialize package to a JSON string; optionally write it to disk."""
        payload = json.dumps(asdict(self), indent=2)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(payload, encoding="utf-8")
        return payload


# ---------------------------------------------------------------------------
# 3. Annex IV completeness checker for CI/CD integration
# Listing 10.2
# ---------------------------------------------------------------------------

def check_annex_iv_completeness(
    package_path: str,
    min_completeness_score: float = 1.0,  # 100% required for production deploys
) -> bool:
    """
    Loads an Annex IV package JSON and checks completeness.
    Exits with code 1 if completeness score is below threshold.
    """
    try:
        data = json.loads(Path(package_path).read_text())
    except FileNotFoundError:
        print(f"ANNEX IV GATE FAILED: Package not found at {package_path}")
        sys.exit(1)

    # Check required top-level fields
    required = [
        "system_name", "system_version", "intended_purpose",
        "model_family", "model_version", "training_data_description",
        "architecture_description", "human_oversight_design",
        "risk_categories_addressed",
    ]

    missing = [f for f in required if not data.get(f)]
    if missing:
        print(f"ANNEX IV GATE FAILED: Missing required fields: {missing}")
        sys.exit(1)

    # Check that evaluation artifacts actually exist on disk
    artifact_fields = ["red_team_report_path", "evaluation_results_path"]
    missing_artifacts = [
        f for f in artifact_fields if not Path(data.get(f, "")).exists()
    ]
    if missing_artifacts:
        print(f"ANNEX IV GATE FAILED: Referenced artifacts not found on disk: {missing_artifacts}")
        sys.exit(1)

    score = 1.0  # every required field present and every referenced artifact exists
    if score < min_completeness_score:
        print(
            f"ANNEX IV GATE FAILED: Completeness score {score:.2%} "
            f"below threshold {min_completeness_score:.2%}"
        )
        sys.exit(1)

    print(f"ANNEX IV GATE PASSED: {package_path} is complete (score {score:.2%}).")
    return True


# ---------------------------------------------------------------------------
# 4. Output Provenance Recorder (HMAC-SHA256)
# Listing 10.3
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceRecord:
    """
    A tamper-evident record linking an LLM output to its input context,
    model identity, and runtime parameters. Only hashes of the input and
    output are stored, never the raw text, which keeps records small,
    storable for years, and free of PII.
    """
    record_id: str
    timestamp_utc: float
    model_id: str
    model_version: str
    prompt_template_version: str
    user_session_id: str          # anonymized session identifier
    retrieval_context_hash: str   # SHA-256 of retrieved documents, if RAG
    input_hash: str               # SHA-256 of the user input
    output_hash: str              # SHA-256 of the model output
    signature: str                # HMAC-SHA256 of the record


def _provenance_payload(record: "ProvenanceRecord") -> Dict[str, Any]:
    """Canonical field set that gets signed / re-verified (excludes `signature` itself)."""
    return {
        "record_id": record.record_id,
        "timestamp_utc": record.timestamp_utc,
        "model_id": record.model_id,
        "model_version": record.model_version,
        "prompt_template_version": record.prompt_template_version,
        "user_session_id": record.user_session_id,
        "retrieval_context_hash": record.retrieval_context_hash,
        "input_hash": record.input_hash,
        "output_hash": record.output_hash,
    }


def create_provenance_record(
    model_id: str,
    model_version: str,
    prompt_template_version: str,
    session_id: str,
    user_input: str,
    model_output: str,
    retrieval_context: str = "",
    signing_key: bytes = None,
) -> ProvenanceRecord:
    """Creates a cryptographically signed provenance record for a model output."""
    if signing_key is None:
        signing_key = os.environ.get("PROVENANCE_SIGNING_KEY", "dev-key").encode()

    import uuid

    record = ProvenanceRecord(
        record_id=str(uuid.uuid4()),
        timestamp_utc=time.time(),
        model_id=model_id,
        model_version=model_version,
        prompt_template_version=prompt_template_version,
        user_session_id=session_id,
        retrieval_context_hash=hashlib.sha256(retrieval_context.encode("utf-8")).hexdigest(),
        input_hash=hashlib.sha256(user_input.encode("utf-8")).hexdigest(),
        output_hash=hashlib.sha256(model_output.encode("utf-8")).hexdigest(),
        signature="",
    )
    canonical = json.dumps(_provenance_payload(record), sort_keys=True)
    record.signature = hmac.new(signing_key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return record


def verify_provenance_record(record: ProvenanceRecord, signing_key: bytes = None) -> bool:
    """Recomputes the HMAC-SHA256 signature and compares it to the stored one."""
    if signing_key is None:
        signing_key = os.environ.get("PROVENANCE_SIGNING_KEY", "dev-key").encode()
    canonical = json.dumps(_provenance_payload(record), sort_keys=True)
    expected = hmac.new(signing_key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, record.signature)


class ProvenanceLog:
    """Appends signed ProvenanceRecords to a JSONL file and reloads them by session/timestamp."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ProvenanceRecord) -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")

    def load_all(self) -> List[ProvenanceRecord]:
        records: List[ProvenanceRecord] = []
        if not self.log_path.exists():
            return records
        with open(self.log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(ProvenanceRecord(**json.loads(line)))
        return records


# ---------------------------------------------------------------------------
# 5. TamperEvidentAuditLog — chained-hash append-only log
# Listing 10.4
# ---------------------------------------------------------------------------

class TamperEvidentAuditLog:
    """
    Append-only log where each entry includes the hash of the previous entry.
    Tampering with any entry breaks the hash chain.
    """

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text("")
        self._last_hash = self._compute_last_hash()

    def _compute_last_hash(self) -> str:
        """
        Returns the previous entry's own hash so a freshly instantiated
        log (a new process re-opening an existing file) resumes the chain
        correctly. Hashing the file's full text here instead would produce
        a value that never matches any entry's stored hash, and the very
        next append() would silently start an unverifiable chain.
        """
        lines = [line for line in self.log_path.read_text().splitlines() if line.strip()]
        if not lines:
            return "genesis"
        return json.loads(lines[-1])["hash"]

    def append(self, event_type: str, payload: dict) -> str:
        """Appends a tamper-evident entry. Returns the entry hash."""
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "payload": payload,
            "prev_hash": self._last_hash,
        }
        entry_json = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()
        entry["hash"] = entry_hash

        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        self._last_hash = entry_hash
        return entry_hash

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Replays every entry, recomputing each entry's hash from its own
        fields, and checks that `prev_hash` matches the previous entry's
        stored hash. Returns {"status": "intact", ...} when the chain is
        unbroken, or {"status": "broken", "broken_at_entry": N} pointing at
        the first entry whose hash no longer matches what the chain expects.
        """
        if not self.log_path.exists() or not self.log_path.read_text().strip():
            return {"status": "intact", "entries_verified": 0}

        prev_hash = "genesis"
        idx = -1
        for idx, line in enumerate(self.log_path.read_text().splitlines()):
            if not line.strip():
                continue
            entry = json.loads(line)
            stored_hash = entry.pop("hash")
            recomputed = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()

            if entry["prev_hash"] != prev_hash or recomputed != stored_hash:
                return {"status": "broken", "broken_at_entry": idx}

            prev_hash = stored_hash

        return {"status": "intact", "entries_verified": idx + 1}


# ---------------------------------------------------------------------------
# 6. NIST AI 600-1 triage report generator (section 10.5.1)
# Listing 10.4b
# ---------------------------------------------------------------------------

class ClusterPriority(str, Enum):
    HIGH = "high"       # address before production launch
    MEDIUM = "medium"   # address in first post-launch sprint
    LOW = "low"         # address in quarterly review


@dataclass
class NistAction:
    action_id: str
    cluster: str
    description: str
    engineering_artifact: str
    book_reference: str
    implemented: bool = False
    evidence_path: str = ""


@dataclass
class NistTriageReport:
    """Generates a prioritized NIST AI 600-1 action list for a given deployment."""
    deployment_type: str  # "chat", "rag", "agent", or "batch"
    risk_level: str       # "low", "medium", "high"
    actions: List[NistAction] = field(default_factory=list)

    def add_action(self, action: NistAction) -> None:
        self.actions.append(action)

    def mark_implemented(self, action_id: str, evidence_path: str = "") -> None:
        for a in self.actions:
            if a.action_id == action_id:
                a.implemented = True
                a.evidence_path = evidence_path
                return
        raise KeyError(f"Unknown NIST action_id: {action_id!r}")

    def gap_summary(self) -> Dict[str, Any]:
        """Returns the sprint backlog: every action not yet implemented."""
        gaps = [asdict(a) for a in self.actions if not a.implemented]
        return {
            "deployment_type": self.deployment_type,
            "risk_level": self.risk_level,
            "total_actions": len(self.actions),
            "implemented_count": len(self.actions) - len(gaps),
            "gap_count": len(gaps),
            "gaps": gaps,
        }


# ---------------------------------------------------------------------------
# 7. NIST AI 600-1 control tracker with gap report generator (section 10.5.3)
# Listing 10.5
# ---------------------------------------------------------------------------

class ImplementationStatus(str, Enum):
    """Implementation status for a NIST AI RMF control."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"


@dataclass
class NISTControl:
    control_id: str
    risk_category: str
    description: str
    status: ImplementationStatus = ImplementationStatus.NOT_STARTED
    owner: str = ""
    evidence_path: str = ""
    notes: str = ""


class NISTAI6001Tracker:
    """Tracks implementation status of NIST AI 600-1 mitigation actions."""

    def __init__(self) -> None:
        self.controls: Dict[str, NISTControl] = {}

    def add_control(self, control: NISTControl) -> None:
        self.controls[control.control_id] = control

    def update_status(
        self,
        control_id: str,
        status: ImplementationStatus,
        owner: str = "",
        evidence_path: str = "",
        notes: str = "",
    ) -> None:
        """Update the implementation status of a control."""
        if control_id not in self.controls:
            raise KeyError(f"Unknown control_id: {control_id!r}")
        ctrl = self.controls[control_id]
        ctrl.status = status
        if owner:
            ctrl.owner = owner
        if evidence_path:
            ctrl.evidence_path = evidence_path
        if notes:
            ctrl.notes = notes

    def gap_report(self) -> Dict[str, Any]:
        """Coverage percentage plus the list of controls not yet implemented or verified."""
        by_status: Dict[str, List[Dict[str, Any]]] = {s.value: [] for s in ImplementationStatus}
        for ctrl in self.controls.values():
            by_status[ctrl.status.value].append(asdict(ctrl))

        total = len(self.controls)
        done = (
            len(by_status[ImplementationStatus.IMPLEMENTED.value])
            + len(by_status[ImplementationStatus.VERIFIED.value])
        )
        coverage_percentage = round(100 * done / total, 1) if total else 0.0

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_controls": total,
            "coverage_percentage": coverage_percentage,
            "summary": {s: len(entries) for s, entries in by_status.items()},
            "gaps": (
                by_status[ImplementationStatus.NOT_STARTED.value]
                + by_status[ImplementationStatus.IN_PROGRESS.value]
            ),
            "implemented": by_status[ImplementationStatus.IMPLEMENTED.value],
            "verified": by_status[ImplementationStatus.VERIFIED.value],
        }

    def to_json(self, path: Optional[Path] = None) -> str:
        """Export tracker state to JSON. Auditors get this file plus the artifact registry."""
        payload = json.dumps({cid: asdict(c) for cid, c in self.controls.items()}, indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(payload, encoding="utf-8")
        return payload


# ---------------------------------------------------------------------------
# 8. Dual-Framework Mapping Report Generator (section 10.6)
# Listing 10.6
# ---------------------------------------------------------------------------

ANNEX_IV_TO_NIST_MAPPING: Dict[str, List[str]] = {
    "1_general_description": ["Information Security", "Value Chain"],
    "2_system_elements": ["Information Security", "Value Chain", "Environmental Risk"],
    "3_monitoring_control": ["Information Security", "CBRN Content", "Human-AI Config"],
    "4_risk_management": [
        "Confabulation", "Data Privacy", "Harmful Bias",
        "Information Security", "Obscene Content",
    ],
    "5_lifecycle_changes": ["Information Security", "Value Chain"],
    "6_conformity_assessment": ["Information Security", "Human-AI Config"],
}

# Practical "ready for initial audit" threshold, not a legal standard.
# Adjust per your organization's risk posture (see section 10.6).
DUAL_FRAMEWORK_COMPLIANT_THRESHOLD = 0.80  # 80% NIST coverage


@dataclass
class DualFrameworkReport:
    system_name: str
    system_version: str
    annex_iv_completeness: dict
    nist_coverage: dict
    cross_references: dict
    generated_at: str


def generate_dual_framework_report(
    annex_iv_package: AnnexIVPackage,
    nist_tracker: NISTAI6001Tracker,
    output_path: str,
) -> str:
    """Produces a single integrated compliance report for EU AI Act + NIST AI RMF."""
    annex_completeness = annex_iv_package.completeness_score()
    nist_gap = nist_tracker.gap_report()

    coverage_fraction = nist_gap["coverage_percentage"] / 100
    # Engineering threshold only — never surface this as a legal determination.
    overall_status = (
        "COMPLIANT"
        if annex_completeness >= 1.0 and coverage_fraction >= DUAL_FRAMEWORK_COMPLIANT_THRESHOLD
        else "IN_PROGRESS"
    )

    report = DualFrameworkReport(
        system_name=annex_iv_package.system_name,
        system_version=annex_iv_package.system_version,
        annex_iv_completeness={
            "score": annex_completeness,
            "missing_required_fields": annex_iv_package.missing_required_fields(),
            "missing_or_stale_artifacts": annex_iv_package.missing_or_stale_artifacts(),
        },
        nist_coverage=nist_gap,
        cross_references=ANNEX_IV_TO_NIST_MAPPING,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    payload = asdict(report)
    payload["overall_status"] = overall_status

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# 9. PostMarketMonitoringReport (section 10.7.4)
# Listing 10.9
# ---------------------------------------------------------------------------

@dataclass
class PostMarketMonitoringReport:
    """EU AI Act post-market monitoring report for a deployment period."""
    system_name: str
    system_version: str
    reporting_period_start: str
    reporting_period_end: str
    total_queries: int
    hallucination_rate: float
    pii_detection_rate: float
    bias_gap_max: float
    incidents_p0: int
    incidents_p1: int
    incidents_p2: int
    serious_incidents_reported: int     # P0 incidents reported to authority
    monitoring_metrics: list = field(default_factory=list)
    trend_alerts: list = field(default_factory=list)

    def requires_regulatory_notification(self) -> bool:
        """True if any P0 incidents occurred that require Article 73 notification."""
        return self.incidents_p0 > 0

    def to_json(self, path: str) -> str:
        data = {
            "monitoring_report": {
                "system": self.system_name,
                "version": self.system_version,
                "period": {
                    "start": self.reporting_period_start,
                    "end": self.reporting_period_end,
                },
                "traffic": {"total_queries": self.total_queries},
                "quality_metrics": {
                    "hallucination_rate": self.hallucination_rate,
                    "pii_detection_rate": self.pii_detection_rate,
                    "bias_gap_max": self.bias_gap_max,
                },
                "incidents": {
                    "p0": self.incidents_p0,
                    "p1": self.incidents_p1,
                    "p2": self.incidents_p2,
                    "serious_incidents_reported": self.serious_incidents_reported,
                },
                "monitoring_metrics": self.monitoring_metrics,
                "trend_alerts": self.trend_alerts,
                "requires_regulatory_notification": self.requires_regulatory_notification(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        result = json.dumps(data, indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(result, encoding="utf-8")
        return result


# ---------------------------------------------------------------------------
# 10. CI/CD Annex IV completeness gate (section 10.9, merge-blocking signal #10)
# Listing 10.12
# ---------------------------------------------------------------------------

def annex_iv_ci_gate(
    annex_iv_package_path: str,
    current_model_version: str,
    max_artifact_age_days: int = 30,
) -> bool:
    """
    Blocks deployment if:
    1. Annex IV package is missing or incomplete
    2. Package references a different model version
    3. Evaluation artifacts are older than max_artifact_age_days
    """
    if not Path(annex_iv_package_path).exists():
        print(f"ANNEX IV GATE FAILED: Package not found at {annex_iv_package_path}")
        sys.exit(1)

    package = json.loads(Path(annex_iv_package_path).read_text())
    failures = []

    # Check 1: Model version matches current deployment
    if package.get("system_version") != current_model_version:
        failures.append(
            f"Package version '{package.get('system_version')}' "
            f"!= current version '{current_model_version}'"
        )

    # Check 2: Required fields present
    missing = [f for f in ANNEX_IV_REQUIRED_FIELDS if not package.get(f)]
    if missing:
        failures.append(f"Missing required fields: {missing}")

    # Check 3: Referenced artifacts exist on disk and aren't stale
    now = time.time()
    for artifact_field in ANNEX_IV_ARTIFACT_PATH_FIELDS:
        artifact_path = package.get(artifact_field, "")
        if not artifact_path or not Path(artifact_path).exists():
            failures.append(f"Artifact not found on disk: {artifact_field}")
            continue
        age_days = (now - os.path.getmtime(artifact_path)) / 86400
        if age_days > max_artifact_age_days:
            failures.append(
                f"Artifact stale: {artifact_field} is {age_days:.1f} days old "
                f"(max {max_artifact_age_days})"
            )

    if failures:
        print(f"ANNEX IV GATE FAILED for {annex_iv_package_path}:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print(f"ANNEX IV GATE PASSED: {annex_iv_package_path} matches version {current_model_version}.")
    return True



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
# Listing 10.7
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
# Listing 10.8
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
# Listing 10.10
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
# Listing 10.11
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
        Level 1-2 without an accompanying tripwire, maps here.

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
# ---------------------------------------------------------------------------
# __main__ — demonstrate all components end-to-end
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("Chapter 10 — EU AI Act + NIST AI RMF Compliance Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="ch10_demo_") as tmpdir:
        tmp = Path(tmpdir)

        # --- 0. Seed placeholder artifact files the index/package check for ---
        for name in (
            "red-team-report.pdf", "bias-assessment.md",
            "evaluation-results.json", "adversarial-robustness.json",
        ):
            (tmp / name).write_text("placeholder artifact content")

        # --- 1. Annex IV artifact index (Listing 10.0) ---
        print("\n--- Annex IV Artifact Index ---")
        deployment_config = {
            "red_team_report_path": str(tmp / "red-team-report.pdf"),
            "evaluation_results_path": str(tmp / "evaluation-results.json"),
        }
        index = generate_annex_iv_index(deployment_config)
        print(f"Ready for audit: {index['ready_for_audit']}")
        print(f"Sections with gaps: {index['sections_with_gaps']}")

        # --- 2. Build a complete Annex IV package (Listing 10.1) ---
        pkg = AnnexIVPackage(
            system_name="CustomerCareBot",
            system_version="2.1.0",
            intended_purpose="Automated tier-1 customer support for retail banking",
            deployment_date="2026-01-15",
            operator_name="Acme Financial AI Ltd.",
            model_family="GPT-4",
            model_version="2024-08-06",
            training_data_description="Fine-tuned on 2.3M anonymized support transcripts (2021-2023).",
            architecture_description="RAG pipeline over policy documents with a GPT-4 generation layer.",
            components=["retrieval-service", "generation-service", "guardrail-filter"],
            monitoring_metrics=["hallucination_rate", "pii_detection_rate", "latency_p99"],
            alert_thresholds={"hallucination_rate": 0.05, "pii_detection_rate": 0.01},
            human_oversight_design="Agents can escalate any conversation; override available at all times.",
            risk_categories_addressed=["Confabulation", "Data Privacy", "Human-AI Config"],
            red_team_report_path=str(tmp / "red-team-report.pdf"),
            bias_assessment_path=str(tmp / "bias-assessment.md"),
            adversarial_robustness_path=str(tmp / "adversarial-robustness.json"),
            evaluation_results_path=str(tmp / "evaluation-results.json"),
        )
        print(f"\nAnnex IV completeness: {pkg.completeness_score():.2%}")
        print(f"Missing required fields: {pkg.missing_required_fields()}")

        # --- 3. Run CI gate (Listing 10.2) ---
        print("\n--- Annex IV CI Gate ---")
        package_path = tmp / "annex-iv-package.json"
        pkg.to_json(package_path)
        check_annex_iv_completeness(str(package_path))

        # --- 3b. Merge-blocking CI/CD gate with version + freshness checks (Listing 10.12) ---
        print("\n--- Annex IV CI/CD Merge Gate ---")
        annex_iv_ci_gate(str(package_path), current_model_version=pkg.system_version)

        # --- 4. Output provenance recorder (Listing 10.3) ---
        print("\n--- Output Provenance Recorder ---")
        record = create_provenance_record(
            model_id="gpt-4o",
            model_version="2024-08-06",
            prompt_template_version="v3.1",
            session_id="sess-abc123",
            user_input="What is my account balance?",
            model_output="Your current balance is EUR 1,240.00.",
            signing_key=b"dev-only-signing-key",
        )
        print(f"Provenance record ID: {record.record_id}")
        print(f"Signature valid: {verify_provenance_record(record, signing_key=b'dev-only-signing-key')}")
        record.output_hash = "tampered_hash"
        print(
            "Signature valid after tamper: "
            f"{verify_provenance_record(record, signing_key=b'dev-only-signing-key')}"
        )

        # --- 5. TamperEvidentAuditLog (Listing 10.4) ---
        print("\n--- Tamper-Evident Audit Log ---")
        log = TamperEvidentAuditLog(str(tmp / "audit.jsonl"))
        log.append("model_deployment", {"version": "2.1.0", "deployed_by": "mlops-pipeline"})
        log.append("policy_update", {"policy": "rate_limit", "new_value": 100})
        log.append("incident_detected", {"severity": "low", "description": "Unusual prompt pattern"})
        print(f"Integrity check: {log.verify_integrity()}")

        # --- 6. NIST AI 600-1 triage report (Listing 10.4b) ---
        print("\n--- NIST AI 600-1 Triage Report ---")
        triage = NistTriageReport(deployment_type="rag", risk_level="high")
        triage.add_action(NistAction(
            action_id="output-validation-1", cluster="Output validation",
            description="Maintain a golden evaluation dataset",
            engineering_artifact="Golden eval dataset + CI job",
            book_reference="Ch 2",
        ))
        triage.add_action(NistAction(
            action_id="human-oversight-1", cluster="Human oversight design",
            description="Build a kill switch reachable within five minutes",
            engineering_artifact="Kill switch + on-call runbook",
            book_reference="Section 10.3.1",
        ))
        triage.mark_implemented("output-validation-1", evidence_path=str(tmp / "evaluation-results.json"))
        gap_summary = triage.gap_summary()
        print(f"Implemented: {gap_summary['implemented_count']}/{gap_summary['total_actions']}")
        print(f"Gaps: {[g['action_id'] for g in gap_summary['gaps']]}")

        # --- 7. NIST AI 600-1 control tracker (Listing 10.5) ---
        print("\n--- NIST AI 600-1 Control Tracker ---")
        tracker = NISTAI6001Tracker()
        tracker.add_control(NISTControl(control_id="GV-1.1", risk_category="Govern",
                                         description="Policies for organizational TEVV"))
        tracker.add_control(NISTControl(control_id="MS-1.1", risk_category="Measure",
                                         description="AI system risks are identified and assessed"))
        tracker.add_control(NISTControl(control_id="MG-4.1", risk_category="Manage",
                                         description="Post-deployment risks are monitored"))
        tracker.update_status("GV-1.1", ImplementationStatus.VERIFIED,
                               owner="AI Governance Team", evidence_path="governance-policy-v3.pdf")
        tracker.update_status("MS-1.1", ImplementationStatus.IN_PROGRESS, owner="Risk Team")
        gap = tracker.gap_report()
        print(f"Coverage: {gap['coverage_percentage']}%")
        print(f"Summary: {gap['summary']}")

        # --- 8. Dual-framework mapping report (Listing 10.6) ---
        print("\n--- Dual-Framework Mapping Report ---")
        dual_report_path = tmp / "compliance-report.json"
        dual_json = generate_dual_framework_report(pkg, tracker, str(dual_report_path))
        dual_report = json.loads(dual_json)
        print(f"Overall status: {dual_report['overall_status']}")
        print(f"NIST coverage: {dual_report['nist_coverage']['coverage_percentage']}%")

        # --- 9. PostMarketMonitoringReport (Listing 10.9) ---
        print("\n--- Post-Market Monitoring Report ---")
        pmm = PostMarketMonitoringReport(
            system_name="CustomerCareBot",
            system_version="2.1.0",
            reporting_period_start="2026-07-01",
            reporting_period_end="2026-07-31",
            total_queries=148_320,
            hallucination_rate=0.031,
            pii_detection_rate=0.004,
            bias_gap_max=0.06,
            incidents_p0=0,
            incidents_p1=1,
            incidents_p2=3,
            serious_incidents_reported=0,
            trend_alerts=["hallucination rate stable over trailing 30 days"],
        )
        print(f"Requires regulatory notification: {pmm.requires_regulatory_notification()}")

        # --- 10. IncidentEscalation (Listing 10.7) ---
        print("\n--- Incident Escalation Pipeline (Article 73 SLA) ---")
        escalation = IncidentEscalation(
            incident_id="INC-2026-0142",
            agent_name="CustomerCareBot",
        )
        escalation.advance(EscalationPhase.ACKNOWLEDGED, completed_by="oncall-jsmith")
        escalation.advance(
            EscalationPhase.CLASSIFIED,
            completed_by="oncall-jsmith",
            notes="Confirmed WRITE_WITHOUT_READ tripwire; reversible impact.",
            is_article_73_candidate=True,
        )
        sla = escalation.sla_status()
        print(f"Current phase: {escalation.current_phase.value}")
        print(f"Is Article 73 candidate: {escalation.is_article_73_candidate}")
        print(f"Article 73 deadline: {sla['article_73_deadline']}")
        print(f"Legal review SLA state: {sla['legal_reviewed']['state']}")

        # --- 11. Article73NotificationPackage (Listing 10.8) ---
        print("\n--- Article 73 Notification Package ---")
        now = time.time()
        notification = Article73NotificationPackage(
            incident_id="INC-2026-0142",
            awareness_timestamp=now,
            incident_timestamp=now - 300,
            agent_name="CustomerCareBot",
            system_description="Tier-1 customer support agent with RAG document access",
            affected_users_count=0,
            incident_description="Agent wrote to an unread backup export path after a prompt injection.",
            initial_impact_assessment="No confirmed data loss; write did not reach persistent storage.",
            containment_actions=["fallback_routing_enabled", "credentials_revoked"],
        )
        print(f"72h package complete: {notification.is_72h_complete()}")
        print(f"15-day package complete: {notification.is_15d_complete()}")
        print(f"Days until deadline: {notification.days_until_deadline():.2f}")
        notification.root_cause = "Prompt injection in retrieved document overrode task scope."
        notification.corrective_measures = ["Added scope-check tripwire", "Tightened tool allowlist"]
        notification.monitoring_improvements = ["Added WRITE_WITHOUT_READ tripwire to CI regression suite"]
        print(f"15-day package complete after follow-up: {notification.is_15d_complete()}")

        # --- 12. ContainmentRunbook (Listing 10.10), with a simulated partial failure ---
        print("\n--- Containment Runbook (simulated partial failure) ---")

        class _MockFallbackRouter:
            def enable_fallback(self, agent_id: str) -> None:
                pass  # succeeds

        class _MockMemoryStore:
            def freeze(self, agent_id: str) -> None:
                raise RuntimeError("memory store unreachable")  # simulated failure

        class _MockCredentialManager:
            def revoke(self, agent_id: str) -> None:
                pass  # succeeds

        class _MockStateCapture:
            def snapshot(self, agent_id: str, incident_type: str) -> None:
                pass  # succeeds

        runbook = ContainmentRunbook(
            fallback_router=_MockFallbackRouter(),
            memory_store=_MockMemoryStore(),
            credential_manager=_MockCredentialManager(),
            state_capture=_MockStateCapture(),
        )
        containment_result = runbook.execute_containment(
            agent_id="agent-cc-014", incident_type="WRITE_WITHOUT_READ"
        )
        print(f"Steps completed: {containment_result.steps_completed}")
        print(f"Steps failed: {containment_result.steps_failed}")
        print(f"Fully contained: {containment_result.fully_contained}")

        # --- 13. IncidentCapture (Listing 10.11) ---
        print("\n--- Incident State Capture ---")
        capture = IncidentCapture(incident_dir=str(tmp / "incidents"))
        incident_id = capture.capture(
            session_id="sess-abc123",
            severity="P1",
            tripwire_event={
                "rule_name": "WRITE_WITHOUT_READ",
                "context": {"resource": "/exports/backup.txt"},
                "timestamp": time.time(),
            },
            action_log=[
                {"step": 1, "action": "read_document", "resource": "/docs/invoice-001.pdf"},
                {"step": 2, "action": "write_classification", "resource": "/exports/backup.txt"},
            ],
            memory_segments=[{"segment": "task_context", "content": "Summarize invoice batch"}],
            agent_name="CustomerCareBot",
        )
        print(f"Incident capture written with ID: {incident_id}")

        # --- 14. IncidentClassifier (sections 10.7-10.8) ---
        print("\n--- Incident Classifier ---")
        classifier = IncidentClassifier()
        classification_input = ClassificationInput(
            tripwire_fired=True,
            tripwire_rule="WRITE_WITHOUT_READ",
            action_was_irreversible=False,
            scope_violation_confirmed=False,
        )
        classification = classifier.classify(classification_input)
        print(f"Severity: {classification.severity}")
        print(f"Rationale: {classification.rationale}")
        print(f"Article 73 clock starts: {classification.article_73_clock_starts}")
        print(f"Resolve within: {classification.resolve_within_hours}h")

    print("\n" + "=" * 70)
    print("All Chapter 10 components demonstrated successfully.")
    print("=" * 70)

