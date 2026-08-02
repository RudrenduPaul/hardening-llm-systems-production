"""
Regression test for validate_output()'s violation-extraction logic
(Listing 9.6, section 9.4).

Guards against a silent no-op regression: Guardrails AI's real
ValidationSummary objects expose `validator_status` and `failure_reason`,
not `failure_level` / `error_message`. A defensive getattr() against the
wrong field names always evaluates to a no-match, which silently produces
an empty `violations` list even when a validator genuinely fails -- no
exception, no warning, just quietly wrong output. This test fabricates a
fake Guardrails outcome with one failing validation_summary entry and
asserts that `violations` is non-empty, so that class of bug fails loudly.

Run with: python3 -m pytest test_ch09_scripts.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import ch09_scripts as ch09


def _fake_openai_client(reply_text: str = "some model output") -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply_text))]
    )
    return client


def test_validate_output_surfaces_failing_validation_summary():
    """A validation_summaries entry with validator_status == 'fail' must
    show up in ContentValidationResult.violations."""
    fake_summary = SimpleNamespace(
        validator_name="ToxicLanguage",
        validator_status="fail",
        property_path="$.output",
        failure_reason="Toxic language detected in sentence: 'you people are...'",
        error_spans=[],
    )
    fake_outcome = SimpleNamespace(
        validation_passed=False,
        validated_output=None,
        validation_summaries=[fake_summary],
        reask_count=1,
    )
    fake_guard = MagicMock()
    fake_guard.parse.return_value = fake_outcome

    result = ch09.validate_output(
        prompt="irrelevant",
        guard=fake_guard,
        client=_fake_openai_client(),
    )

    assert result.violations, "violations must be non-empty when a validator fails"
    assert "Toxic language detected" in result.violations[0]
    assert result.passed is False
    assert result.reask_count == 1


def test_validate_output_ignores_passing_validation_summary():
    """A validation_summaries entry with validator_status == 'pass' must
    NOT show up in violations."""
    fake_summary = SimpleNamespace(
        validator_name="DetectPII",
        validator_status="pass",
        property_path="$.output",
        failure_reason=None,
        error_spans=[],
    )
    fake_outcome = SimpleNamespace(
        validation_passed=True,
        validated_output="clean output",
        validation_summaries=[fake_summary],
        reask_count=0,
    )
    fake_guard = MagicMock()
    fake_guard.parse.return_value = fake_outcome

    result = ch09.validate_output(
        prompt="irrelevant",
        guard=fake_guard,
        client=_fake_openai_client(),
    )

    assert result.violations == []
    assert result.passed is True


if __name__ == "__main__":
    test_validate_output_surfaces_failing_validation_summary()
    test_validate_output_ignores_passing_validation_summary()
    print("All ch09_scripts tests passed.")
