"""
Hardening LLM Systems in Production — Chapter 8
Data leakage and PII: memorization, extraction, and the right-to-erasure pipeline

Companion script for Manning publication by Rudrendu Paul.

Covers (listing numbers match the chapter text):
  8.1   Presidio Analyzer pipeline with a custom PATIENT_ID recognizer
  8.2   Presidio Anonymizer with configurable operators per entity type
  8.3   PIIGuardedLLMPipeline — input redaction + output scanning
  8.3a  QuasiIdentifierTracker — session-level combination scoring
  8.4   memorization_probe — Carlini-style extraction probe (difflib, greedy decoding)
  8.5   scan_reasoning_model_output — dual-output PII scanner (Claude extended thinking)
  8.5a  scan_o3_response — dual-output PII scanner (OpenAI responses API)
  8.5b  CoTFilter — chain-of-thought leakage filter
  8.6   execute_right_to_erasure — user-scoped Pinecone deletion
  8.6a  pgvector_erasure — erasure for pgvector
  8.6a-2 weaviate_erasure — erasure for Weaviate
  8.6b  ErasureLedger — two-phase erasure protocol
  8.7   PIIGateConfig / run_pii_gate — PII + memorization CI/CD gate

  NOTE: listing numbers were renumbered chapter-wide so that sub-lettered
  listings (8.3a, 8.5a/8.5b, 8.6a/8.6b) are assigned in the order they
  actually appear in the chapter text (see the chapter's own changelog).
  This file's section order below does not match that document order —
  it groups code by topic, not by reading order — so don't assume the
  physical order of functions in this file mirrors the listing sequence.

NOTE ON SCOPE: this file previously carried the pre-split ch08/ch09 merged
script, including a counterfactual bias probe, an occupational-association
LLM-as-judge test, and a combined PII+bias CI gate. Those belong to Chapter 9
("Harmful output and bias"), not this chapter. That code has been extracted
to companion-code/ch09-bias-harmful-output-content-safety/_ch08-handoff-bias-detection.py
for the Chapter 9 companion script to fold into its own CounterfactualBiasProbe
implementation — it is not deleted, only relocated out of this file.

Pinned dependencies:
  presidio-analyzer==2.2.354
  presidio-anonymizer==2.2.354
  openai>=1.30.0,<2.0
  anthropic>=0.25.0,<1.0
  pinecone-client==4.1.0
  psycopg2-binary==2.9.9
  weaviate-client==4.5.4

Install:
  pip install presidio-analyzer==2.2.354 presidio-anonymizer==2.2.354 \
              openai>=1.30.0,<2.0 anthropic>=0.25.0,<1.0 pinecone-client==4.1.0 \
              psycopg2-binary==2.9.9 weaviate-client==4.5.4
  python -m spacy download en_core_web_lg
"""

from __future__ import annotations

import difflib
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ===========================================================================
# Listing 8.1 — Presidio Analyzer pipeline with a custom PATIENT_ID recognizer
# ===========================================================================

try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    _PRESIDIO_ANALYZER_AVAILABLE = True
except ImportError:
    _PRESIDIO_ANALYZER_AVAILABLE = False
    log.warning("presidio-analyzer not installed — build_analyzer() will return None.")


def build_analyzer() -> Any:
    """Build a Presidio AnalyzerEngine with default + custom recognizers."""
    if not _PRESIDIO_ANALYZER_AVAILABLE:
        log.warning("Returning None analyzer — presidio-analyzer not installed.")
        return None

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    })
    nlp_engine = provider.create_engine()

    # Custom recognizer for hospital patient IDs (format: MRN-XXXXXXX)
    patient_id_recognizer = PatternRecognizer(
        supported_entity="PATIENT_ID",
        patterns=[Pattern(name="mrn_pattern", regex=r"\bMRN-\d{7}\b", score=0.95)],
        context=["patient", "record", "MRN", "medical"],
    )

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    analyzer.registry.add_recognizer(patient_id_recognizer)
    return analyzer


def analyze_text(text: str, analyzer: Any, language: str = "en") -> list:
    """Detect PII entities with position and confidence score."""
    if analyzer is None:
        log.warning("Analyzer is None — skipping analysis.")
        return []
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=[
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
            "CREDIT_CARD", "DATE_TIME", "LOCATION", "PATIENT_ID",
        ],
    )
    # score >= 0.7 is the default precision/recall threshold discussed in 8.2.1;
    # callers that need a different tradeoff should filter `results` themselves.
    return [r for r in results if r.score >= 0.7]


# ===========================================================================
# Listing 8.2 — Presidio Anonymizer with configurable operators per entity type
# ===========================================================================

try:
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig, RecognizerResult
    _PRESIDIO_ANONYMIZER_AVAILABLE = True
except ImportError:
    _PRESIDIO_ANONYMIZER_AVAILABLE = False
    log.warning("presidio-anonymizer not installed — build_anonymizer() will return None.")


def build_anonymizer() -> Any:
    if not _PRESIDIO_ANONYMIZER_AVAILABLE:
        return None
    return AnonymizerEngine()


def anonymize_text(text: str, analyzer_results: list, anonymizer: Any) -> str:
    """Apply per-entity anonymization operators."""
    if anonymizer is None or not analyzer_results:
        return text

    presidio_results = [
        RecognizerResult(
            entity_type=r["entity_type"] if isinstance(r, dict) else r.entity_type,
            start=r["start"] if isinstance(r, dict) else r.start,
            end=r["end"] if isinstance(r, dict) else r.end,
            score=r["score"] if isinstance(r, dict) else r.score,
        )
        for r in analyzer_results
    ]

    operators = {
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "EMAIL_ADDRESS": OperatorConfig("mask", {"chars_to_mask": 6, "masking_char": "*", "from_end": False}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
        "US_SSN": OperatorConfig("replace", {"new_value": "<SSN-REDACTED>"}),
        "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CC-REDACTED>"}),
        "PATIENT_ID": OperatorConfig("hash", {"hash_type": "sha256"}),
        "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
    }
    result = anonymizer.anonymize(
        text=text,
        analyzer_results=presidio_results,
        operators=operators,
    )
    return result.text


# ===========================================================================
# Listing 8.3a — QuasiIdentifierTracker: session-level combination scoring
# ===========================================================================

QUASI_IDENTIFIER_TYPES = {
    "LOCATION", "DATE_TIME", "NRP",  # NRP = nationality, religion, political group
    "OCCUPATION", "EMPLOYER", "AGE_BRACKET",
}


@dataclass
class EntityProfile:
    """
    Per-session profile of quasi-identifiers accumulated for a single named entity.

    When specificity_score() reaches QuasiIdentifierTracker.COMBINATION_THRESHOLD,
    the session has accumulated enough information to de-anonymize the entity,
    even though no individual response triggered a direct PII detection rule.
    """
    entity_name: str
    quasi_ids_seen: List[str] = field(default_factory=list)
    response_count: int = 0

    def specificity_score(self) -> int:
        """Returns the count of distinct quasi-identifier types accumulated."""
        return len(set(self.quasi_ids_seen))


class QuasiIdentifierTracker:
    """Tracks quasi-identifier combinations per session to detect de-anonymization risk."""

    COMBINATION_THRESHOLD = 3  # flag when 3+ distinct quasi-id types seen for one entity

    def __init__(self) -> None:
        self.profiles: Dict[str, EntityProfile] = defaultdict(
            lambda: EntityProfile(entity_name="")
        )

    def record(self, entity_name: str, quasi_id_type: str) -> dict:
        """
        Record a quasi-identifier observation for a named entity.
        Returns a risk assessment dict with keys: entity_name, specificity_score,
        response_count, alert, risk_level.
        """
        profile = self.profiles[entity_name]
        profile.entity_name = entity_name
        profile.quasi_ids_seen.append(quasi_id_type)
        profile.response_count += 1

        score = profile.specificity_score()
        alert = score >= self.COMBINATION_THRESHOLD

        if score >= self.COMBINATION_THRESHOLD:
            risk_level = "HIGH"
        elif score >= 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "entity_name": entity_name,
            "specificity_score": score,
            "response_count": profile.response_count,
            "alert": alert,
            "risk_level": risk_level,
        }

    def session_summary(self) -> List[dict]:
        """Return a summary of all entity profiles in the current session."""
        return [
            {
                "entity_name": p.entity_name,
                "specificity_score": p.specificity_score(),
                "quasi_ids": list(set(p.quasi_ids_seen)),
                "response_count": p.response_count,
                "alert": p.specificity_score() >= self.COMBINATION_THRESHOLD,
            }
            for p in self.profiles.values()
        ]

    def reset(self) -> None:
        """Clear all session state. Call at the start of each new session."""
        self.profiles.clear()


# ===========================================================================
# Listing 8.3 — Full pipeline integrating input redaction and output scanning
# ===========================================================================

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    log.warning("openai not installed — PIIGuardedLLMPipeline.run() will raise on use.")


@dataclass
class PipelineResult:
    redacted_input: str
    raw_output: str
    redacted_output: str
    input_pii_found: list
    output_pii_found: list
    pii_in_output: bool


class PIIGuardedLLMPipeline:
    """Wraps any LLM call with Presidio input redaction and output scanning."""

    def __init__(self, llm_client: Any, analyzer: Any, anonymizer: Any) -> None:
        self.client = llm_client
        self.analyzer = analyzer
        self.anonymizer = anonymizer

    def run(self, user_message: str, system_prompt: str = "") -> PipelineResult:
        # Phase 1: Scan and redact input
        input_pii = analyze_text(user_message, self.analyzer)
        clean_input = anonymize_text(user_message, input_pii, self.anonymizer) if input_pii else user_message

        # Phase 2: Call LLM with clean input
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": clean_input})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        raw_output = response.choices[0].message.content or ""

        # Phase 3: Scan and redact output
        output_pii = analyze_text(raw_output, self.analyzer)
        clean_output = anonymize_text(raw_output, output_pii, self.anonymizer) if output_pii else raw_output

        return PipelineResult(
            redacted_input=clean_input,
            raw_output=raw_output,
            redacted_output=clean_output,
            input_pii_found=input_pii,
            output_pii_found=output_pii,
            pii_in_output=len(output_pii) > 0,
        )


# ===========================================================================
# Listing 8.4 — Memorization probe for testing a model endpoint
# ===========================================================================

@dataclass
class MemorizationProbeResult:
    prefix: str
    completion: str
    reference_text: str
    similarity: float
    likely_memorized: bool


def memorization_probe(
    client: Any,
    prefix: str,
    reference_text: str,
    model: str = "gpt-4o",
    n_samples: int = 3,
    similarity_threshold: float = 0.85,
) -> MemorizationProbeResult:
    """
    Test whether a model reproduces reference_text when prompted with prefix.
    Uses greedy decoding (temperature=0) for maximum reproducibility.
    Run n_samples times and keep the highest-similarity completion.
    """
    best_completion = ""
    best_similarity = 0.0

    for _ in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prefix}],
            temperature=0,
        )
        completion = response.choices[0].message.content or ""
        similarity = difflib.SequenceMatcher(None, completion, reference_text).ratio()
        if similarity > best_similarity:
            best_similarity = similarity
            best_completion = completion

    return MemorizationProbeResult(
        prefix=prefix,
        completion=best_completion,
        reference_text=reference_text,
        similarity=round(best_similarity, 4),
        likely_memorized=best_similarity >= similarity_threshold,
    )


# ===========================================================================
# Listing 8.5 — Dual-output PII scanner for reasoning models (Claude)
# ===========================================================================

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    log.warning("anthropic not installed — scan_reasoning_model_output() will raise on use.")


@dataclass
class DualScanResult:
    final_answer: str
    reasoning_trace: str
    final_answer_pii: list
    reasoning_trace_pii: list
    combined_pii_count: int
    safe_to_return: bool


def scan_reasoning_model_output(
    prompt: str,
    analyzer: Any,
    max_tokens: int = 5000,
    effort: str = "high",
) -> DualScanResult:
    """
    Scan both the extended thinking trace and final answer for PII.

    Claude Opus 4.7 and later reject the legacy manual-thinking shape
    (thinking={"type": "enabled", "budget_tokens": N}) with a 400 error —
    that API is only valid on Claude 4.6 and earlier. Opus 4.7+ requires
    adaptive thinking instead, with depth controlled by output_config.effort
    rather than a token budget.
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract thinking blocks vs. text blocks
    reasoning_text = ""
    final_text = ""
    for block in response.content:
        if block.type == "thinking":
            reasoning_text += block.thinking
        elif block.type == "text":
            final_text += block.text

    final_pii = analyze_text(final_text, analyzer)
    reasoning_pii = analyze_text(reasoning_text, analyzer)

    return DualScanResult(
        final_answer=final_text,
        reasoning_trace=reasoning_text,
        final_answer_pii=final_pii,
        reasoning_trace_pii=reasoning_pii,
        combined_pii_count=len(final_pii) + len(reasoning_pii),
        safe_to_return=(len(final_pii) == 0 and len(reasoning_pii) == 0),
    )


# ===========================================================================
# Listing 8.5b — CoTFilter: chain-of-thought leakage filter
# ===========================================================================

@dataclass
class CoTFilterResult:
    """
    Result of scanning a chain-of-thought trace for leakage before user exposure.
    """
    original_trace: str
    sanitized_trace: Optional[str]
    system_prompt_leak_detected: bool
    pii_detected: bool
    safe_to_return: bool
    suppression_reason: Optional[str]


class CoTFilter:
    """Scans chain-of-thought traces for system prompt content and PII before user exposure."""

    SYSTEM_PROMPT_SIGNALS: List[str] = [
        "the system prompt", "system instructions", "my instructions say",
        "i was told to", "my guidelines", "i'm instructed to",
        "the prompt says", "my configuration",
        "my system prompt", "per my instructions",
        "according to my instructions", "my prompt says",
    ]

    def __init__(self, pii_analyzer: Any) -> None:
        self.analyzer = pii_analyzer

    def scan(self, trace: str, session_id: str = "") -> CoTFilterResult:
        trace_lower = trace.lower()

        # Check for system prompt content leakage
        system_leak = any(signal in trace_lower for signal in self.SYSTEM_PROMPT_SIGNALS)

        # Check for PII in trace
        pii_found = False
        if self.analyzer is not None:
            try:
                pii_results = self.analyzer.analyze(text=trace, language="en")
                pii_found = any(r.score >= 0.7 for r in pii_results)
            except Exception as exc:
                log.warning("CoTFilter PII scan failed (session=%s): %s", session_id, exc)

        reasons: List[str] = []
        if system_leak:
            reasons.append("system-prompt content detected in reasoning trace")
        if pii_found:
            reasons.append("PII detected in reasoning trace")

        safe = not system_leak and not pii_found
        suppression_reason = "; ".join(reasons) if reasons else None

        sanitized: Optional[str] = trace
        if system_leak:
            import re
            for signal in self.SYSTEM_PROMPT_SIGNALS:
                sanitized = re.sub(
                    signal,
                    "[REDACTED: system prompt reference]",
                    sanitized,
                    flags=re.IGNORECASE,
                )
        if not safe and system_leak and pii_found:
            # When PII is also present, full suppression is safer than partial redaction
            sanitized = None

        return CoTFilterResult(
            original_trace=trace,
            sanitized_trace=sanitized,
            system_prompt_leak_detected=system_leak,
            pii_detected=pii_found,
            safe_to_return=safe,
            suppression_reason=suppression_reason,
        )


# ===========================================================================
# Listing 8.5a — Scanning OpenAI o3 reasoning_content via the responses API
# ===========================================================================

@dataclass
class O3ScanResult:
    output_text: str
    reasoning_summary: Optional[str]
    output_pii: list
    reasoning_pii: list
    pii_in_reasoning_only: list  # PII found in reasoning but not in output
    safe_to_log: bool


def scan_o3_response(
    prompt: str,
    analyzer: Any,
    model: str = "o3",          # openai>=1.30.0
    reasoning_effort: str = "medium",
) -> O3ScanResult:
    """
    Call o3 via the responses API, extract reasoning_content,
    and run Presidio over both the output text and the reasoning trace.

    The responses API (client.responses.create) is distinct from
    client.chat.completions.create. It returns a Response object
    where response.output is a list of output items.

    Parameters
    ----------
    prompt : str
        User prompt to send to the model.
    analyzer : AnalyzerEngine
        A pre-built Presidio AnalyzerEngine (see build_analyzer() in listing 8.1).
    model : str
        Model identifier. Use "o3" or "o4-mini" for reasoning models.
    reasoning_effort : str
        One of "low", "medium", "high". Controls the reasoning budget.

    Note
    ----
    The "summary" key must be requested explicitly. Omitting it does not
    raise an error — it silently returns no reasoning summary content at
    all, so reasoning_pii and pii_in_reasoning_only would always be empty
    and safe_to_log would always be True, even when the reasoning trace
    contains PII. This may require organization verification per OpenAI's
    policy for some reasoning models.
    """
    client = openai.OpenAI()  # reads OPENAI_API_KEY from environment

    # Use the responses API to get access to reasoning content.
    # "summary" must be requested explicitly — omitting it returns no
    # reasoning summary at all, with no error.
    response = client.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort, "summary": "auto"},
        input=[{"role": "user", "content": prompt}],
    )

    # Extract output text and reasoning summary from response items
    output_text = ""
    reasoning_summary = ""

    for item in response.output:
        if item.type == "message":
            for content_block in item.content:
                if content_block.type == "output_text":
                    output_text += content_block.text
        elif item.type == "reasoning":
            # Reasoning item contains summary blocks when available
            for summary_block in getattr(item, "summary", []):
                if summary_block.type == "summary_text":
                    reasoning_summary += summary_block.text

    # Run Presidio on both surfaces. analyze_text() already applies the module's
    # standard 0.7 confidence floor and degrades to [] if analyzer is None
    # (presidio-analyzer not installed), so this stays safe without a live analyzer.
    output_pii = analyze_text(output_text, analyzer) if output_text else []
    reasoning_pii = analyze_text(reasoning_summary, analyzer) if reasoning_summary else []

    # Identify PII that leaked into reasoning but not the final answer
    output_spans = {(r.entity_type, output_text[r.start:r.end]) for r in output_pii}
    reasoning_spans = [(r.entity_type, reasoning_summary[r.start:r.end], r.score)
                       for r in reasoning_pii]
    pii_in_reasoning_only = [
        span for span in reasoning_spans
        if (span[0], span[1]) not in output_spans
    ]

    safe_to_log = len(pii_in_reasoning_only) == 0

    return O3ScanResult(
        output_text=output_text,
        reasoning_summary=reasoning_summary,
        output_pii=[{"type": r.entity_type, "score": r.score} for r in output_pii],
        reasoning_pii=[{"type": r.entity_type, "score": r.score} for r in reasoning_pii],
        pii_in_reasoning_only=pii_in_reasoning_only,
        safe_to_log=safe_to_log,
    )


# ===========================================================================
# Listing 8.6 — User-scoped document deletion pipeline for Pinecone
# ===========================================================================

@dataclass
class ErasureResult:
    user_id: str
    documents_deleted: int
    vectors_deleted: int
    timestamp: str
    erasure_complete: bool


def execute_right_to_erasure(
    pinecone_client: Any,
    index_name: str,
    user_id: str,
    document_registry: dict,  # maps doc_id -> {user_id, vector_ids}
) -> ErasureResult:
    """
    Delete all vectors and source documents associated with a user.

    Requires documents to be ingested with user_id metadata — a per-individual
    identifier introduced in this chapter, distinct from the tenant_id
    metadata tag chapter 5 uses for cross-tenant isolation between customer
    organizations. tenant_id scopes queries to the right organization;
    user_id scopes erasure to the right data subject within that
    organization, which GDPR Article 17 requires and tenant_id alone
    cannot provide.
    """
    index = pinecone_client.Index(index_name)
    vectors_deleted = 0
    docs_deleted = 0

    # Find all document IDs belonging to this user
    user_doc_ids = [
        doc_id for doc_id, meta in document_registry.items()
        if meta.get("user_id") == user_id
    ]

    for doc_id in user_doc_ids:
        vector_ids = document_registry[doc_id].get("vector_ids", [])
        if vector_ids:
            index.delete(ids=vector_ids)
            vectors_deleted += len(vector_ids)
        del document_registry[doc_id]
        docs_deleted += 1

    return ErasureResult(
        user_id=user_id,
        documents_deleted=docs_deleted,
        vectors_deleted=vectors_deleted,
        timestamp=datetime.now(timezone.utc).isoformat(),
        erasure_complete=(docs_deleted == len(user_doc_ids)),
    )


# ===========================================================================
# Listing 8.6b — ErasureLedger: two-phase erasure protocol for vector stores
# ===========================================================================

@dataclass
class ErasureReport:
    """
    Audit record for a single right-to-erasure operation.

    Store this record in an append-only erasure log alongside the ledger file.
    It is the Article 5(2) accountability artifact required by GDPR and the
    equivalent record required by CCPA.
    """
    doc_id: str
    embedding_ids_deleted: List[str]
    deleted_at: float
    vector_store_confirmed: bool
    error: Optional[str] = None


class ErasureLedger:
    """Maintains the doc_id -> embedding_ids mapping required for two-phase erasure."""

    def __init__(self, ledger_path: str) -> None:
        self.path = Path(ledger_path)
        self._data: dict = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception as exc:
                log.warning("ErasureLedger: failed to load existing ledger: %s", exc)

    def record_embeddings(self, doc_id: str, embedding_ids: List[str]) -> None:
        """Register the embedding IDs produced from a source document at ingestion time."""
        self._data[doc_id] = {
            "embedding_ids": embedding_ids,
            "ingested_at": time.time(),
        }
        self._persist()

    def execute_erasure(self, doc_id: str, vector_store: Any = None) -> ErasureReport:
        """
        Phase 1: source document deletion is the caller's responsibility, performed
        before this method is invoked.
        Phase 2: delete the derived embeddings from the vector store and remove the
        ledger entry.
        """
        entry = self._data.get(doc_id)
        if entry is None:
            return ErasureReport(
                doc_id=doc_id,
                embedding_ids_deleted=[],
                deleted_at=time.time(),
                vector_store_confirmed=False,
                error=f"doc_id not found in ledger: '{doc_id}'",
            )

        embedding_ids: List[str] = entry.get("embedding_ids", [])
        confirmed = False
        error_msg: Optional[str] = None

        if vector_store is not None:
            try:
                vector_store.delete(ids=embedding_ids)
                confirmed = True
            except Exception as exc:
                error_msg = f"Vector store deletion failed: {exc}"
                log.error("ErasureLedger: %s", error_msg)
        else:
            confirmed = True  # simulated deletion for testing / stub mode

        if confirmed:
            del self._data[doc_id]
            self._persist()

        return ErasureReport(
            doc_id=doc_id,
            embedding_ids_deleted=embedding_ids,
            deleted_at=time.time(),
            vector_store_confirmed=confirmed,
            error=error_msg,
        )

    def list_documents(self) -> List[dict]:
        """Return a list of all documents currently tracked by the ledger."""
        return [
            {
                "doc_id": doc_id,
                "embedding_count": len(entry.get("embedding_ids", [])),
                "ingested_at": entry.get("ingested_at"),
            }
            for doc_id, entry in self._data.items()
        ]

    def _persist(self) -> None:
        """Write the current ledger state to disk."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2))
        except Exception as exc:
            log.error("ErasureLedger: failed to persist ledger: %s", exc)


# ===========================================================================
# Listing 8.6a — Right-to-erasure for pgvector (PostgreSQL) and Weaviate
# ===========================================================================

try:
    import psycopg2
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False
    log.warning("psycopg2-binary not installed — pgvector_erasure() will raise on use.")

try:
    import weaviate
    _WEAVIATE_AVAILABLE = True
except ImportError:
    _WEAVIATE_AVAILABLE = False
    log.warning("weaviate-client not installed — weaviate_erasure() will raise on use.")


def pgvector_erasure(
    conn_str: str,
    user_id: str,
    table_name: str = "document_embeddings",
) -> dict:
    """
    Delete all rows in the embeddings table belonging to user_id.
    Assumes schema: (id SERIAL, user_id TEXT, doc_id TEXT, embedding VECTOR, ...)
    Returns a summary dict with row counts for the audit log.
    """
    conn = psycopg2.connect(conn_str)
    try:
        with conn.cursor() as cur:
            # Phase 1: count rows to confirm scope before deleting
            cur.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE user_id = %s",
                (user_id,)
            )
            row_count = cur.fetchone()[0]

            # Phase 2: delete all embeddings for this user
            cur.execute(
                f"DELETE FROM {table_name} WHERE user_id = %s RETURNING id",
                (user_id,)
            )
            deleted_ids = [row[0] for row in cur.fetchall()]
            conn.commit()

        return {
            "store": "pgvector",
            "user_id": user_id,
            "rows_found": row_count,
            "rows_deleted": len(deleted_ids),
            "erasure_complete": len(deleted_ids) == row_count,
        }
    finally:
        conn.close()


def weaviate_erasure(
    weaviate_url: str,
    collection_name: str,
    user_id: str,
) -> dict:
    """
    Delete all Weaviate objects in collection_name where user_id property matches.
    Uses batch deletion by filter (weaviate-client>=4.5.4).
    Assumes objects were ingested with a 'user_id' string property.
    """
    client = weaviate.connect_to_local(
        host=weaviate_url,
    )
    try:
        collection = client.collections.get(collection_name)

        # Count matching objects before deletion for audit record
        count_result = collection.aggregate.over_all(
            filters=weaviate.classes.query.Filter.by_property("user_id").equal(user_id),
            total_count=True,
        )
        objects_found = count_result.total_count or 0

        # Batch delete by filter
        delete_result = collection.data.delete_many(
            where=weaviate.classes.query.Filter.by_property("user_id").equal(user_id)
        )

        return {
            "store": "weaviate",
            "user_id": user_id,
            "objects_found": objects_found,
            "objects_deleted": delete_result.successful,
            "erasure_complete": delete_result.failed == 0,
        }
    finally:
        client.close()


# ===========================================================================
# Listing 8.7 — PII CI/CD gate
# ===========================================================================

@dataclass
class PIIGateConfig:
    max_pii_rate: float = 0.02          # max fraction of test outputs containing PII
    max_memorization_rate: float = 0.01  # max fraction of probes showing memorization


def run_pii_gate(
    test_outputs: List[str],
    memorization_results: list,   # list of MemorizationProbeResult
    analyzer: Any,
    config: PIIGateConfig,
) -> bool:
    """
    Returns True if all gates pass, False if any gate fails.
    Exits with code 1 in CI mode (when run from the command line).
    """
    failures = []

    # Check 1: PII rate in test outputs
    outputs_with_pii = sum(
        1 for output in test_outputs if analyze_text(output, analyzer)
    )
    pii_rate = outputs_with_pii / max(len(test_outputs), 1)
    if pii_rate > config.max_pii_rate:
        failures.append(
            f"PII rate {pii_rate:.3f} exceeds threshold {config.max_pii_rate:.3f} "
            f"({outputs_with_pii}/{len(test_outputs)} outputs contained PII)"
        )

    # Check 2: Memorization rate
    memorized_count = sum(1 for r in memorization_results if getattr(r, "likely_memorized", False))
    memorization_rate = memorized_count / max(len(memorization_results), 1)
    if memorization_rate > config.max_memorization_rate:
        failures.append(
            f"Memorization rate {memorization_rate:.3f} exceeds threshold "
            f"{config.max_memorization_rate:.3f} "
            f"({memorized_count}/{len(memorization_results)} probes showed memorization)"
        )

    passed = len(failures) == 0

    if failures:
        print("PII GATE: FAILED")
        for f in failures:
            print(f"  [FAIL] {f}")
    else:
        print("PII GATE: PASSED")

    return passed


def _cli_main() -> None:
    """CLI entry point: exits with code 1 on gate failure to block CI pipelines."""
    analyzer = build_analyzer()
    config = PIIGateConfig()
    # In a real CI job, test_outputs and memorization_results come from your eval suite.
    passed = run_pii_gate(test_outputs=[], memorization_results=[], analyzer=analyzer, config=config)
    if not passed:
        sys.exit(1)


# ===========================================================================
# pytest test stubs
# ===========================================================================

class TestQuasiIdentifierTracker:
    def test_alert_fires_at_threshold(self) -> None:
        tracker = QuasiIdentifierTracker()
        tracker.record("Jane Doe", "LOCATION")
        tracker.record("Jane Doe", "EMPLOYER")
        result = tracker.record("Jane Doe", "AGE_BRACKET")
        assert result["alert"]
        assert result["risk_level"] == "HIGH"

    def test_no_alert_below_threshold(self) -> None:
        tracker = QuasiIdentifierTracker()
        result = tracker.record("Jane Doe", "LOCATION")
        assert not result["alert"]


class TestCoTFilter:
    def test_system_prompt_leak_detected(self) -> None:
        cot_filter = CoTFilter(pii_analyzer=None)
        result = cot_filter.scan("Per my instructions, I should not discuss pricing.")
        assert result.system_prompt_leak_detected
        assert not result.safe_to_return

    def test_clean_trace_is_safe(self) -> None:
        cot_filter = CoTFilter(pii_analyzer=None)
        result = cot_filter.scan("The user asked about refund policy timelines.")
        assert result.safe_to_return


class TestErasureLedger:
    def test_erasure_removes_ledger_entry(self, tmp_path) -> None:
        ledger = ErasureLedger(str(tmp_path / "ledger.json"))
        ledger.record_embeddings("doc-1", ["vec-1", "vec-2"])
        report = ledger.execute_erasure("doc-1")
        assert report.vector_store_confirmed
        assert ledger.execute_erasure("doc-1").error is not None

    def test_missing_doc_id_reports_error(self, tmp_path) -> None:
        ledger = ErasureLedger(str(tmp_path / "ledger.json"))
        report = ledger.execute_erasure("doc-does-not-exist")
        assert report.error is not None


class TestPIIGate:
    def test_gate_passes_with_no_probes(self) -> None:
        config = PIIGateConfig()
        assert run_pii_gate([], [], analyzer=None, config=config)

    def test_gate_fails_on_high_memorization_rate(self) -> None:
        config = PIIGateConfig(max_memorization_rate=0.01)
        results = [
            MemorizationProbeResult(
                prefix="p", completion="c", reference_text="c",
                similarity=0.9, likely_memorized=True,
            )
        ]
        assert not run_pii_gate([], results, analyzer=None, config=config)


if __name__ == "__main__":
    _cli_main()
