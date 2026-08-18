"""Lossless first-name normalization and eligibility decisions."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import pandas as pd

NameInput: TypeAlias = str | None
NameCollection: TypeAlias = NameInput | Sequence[NameInput] | pd.Series

_UNICODE_SCRIPT_MARKERS = {
    "BENGALI": "Beng",
    "DEVANAGARI": "Deva",
    "GUJARATI": "Gujr",
    "GURMUKHI": "Guru",
    "KANNADA": "Knda",
    "LATIN": "Latn",
    "MALAYALAM": "Mlym",
    "ORIYA": "Orya",
    "ODIA": "Orya",
    "TAMIL": "Taml",
    "TELUGU": "Telu",
}


@dataclass(frozen=True, slots=True)
class NormalizedFirstName:
    """One normalized input and its eligibility decision."""

    name: str | None
    normalized_name: str | None
    detected_script: str | None
    script_supported: bool | None
    abstention_reason: str | None

    @property
    def abstained(self) -> bool:
        """Return whether the input must not be scored or looked up."""
        return self.abstention_reason is not None


def coerce_name_collection(names: NameCollection) -> list[NameInput]:
    """Convert a supported scalar or one-dimensional collection to a list."""
    if isinstance(names, str) or names is None:
        return [names]
    if isinstance(names, pd.Series):
        values = names.tolist()
    elif isinstance(names, Sequence) and not isinstance(names, (bytes, bytearray)):
        values = list(names)
    else:
        raise TypeError("names must be a string, None, Series, or sequence of strings")

    for position, value in enumerate(values):
        if not _is_missing(value) and not isinstance(value, str):
            raise TypeError(f"names[{position}] must be a string or missing")
    return [None if _is_missing(value) else value for value in values]


def normalize_for_model(name: NameInput) -> NormalizedFirstName:
    """Normalize one name and enforce the learned model's frozen input domain."""
    normalized = _normalize_common(name)
    if normalized.abstention_reason == "missing-name":
        return normalized

    script_decision = _with_script_decision(normalized, frozenset({"Latn"}))
    if normalized.abstention_reason is not None:
        return NormalizedFirstName(
            normalized.name,
            normalized.normalized_name,
            script_decision.detected_script,
            script_decision.script_supported,
            normalized.abstention_reason,
        )
    if script_decision.abstention_reason is not None:
        return script_decision

    if normalized.name is None or normalized.normalized_name is None:
        raise RuntimeError(
            "eligible normalization unexpectedly produced a missing name"
        )
    source_without_outer_whitespace = unicodedata.normalize(
        "NFC", normalized.name
    ).strip()
    if not source_without_outer_whitespace.isascii() or not all(
        character.isalpha() for character in source_without_outer_whitespace
    ):
        return _with_reason(script_decision, "unsupported-characters")
    if (
        not normalized.normalized_name.isascii()
        or not normalized.normalized_name.isalpha()
    ):
        return _with_reason(script_decision, "unsupported-characters")
    if any(
        first == second == third
        for first, second, third in zip(
            normalized.normalized_name,
            normalized.normalized_name[1:],
            normalized.normalized_name[2:],
            strict=False,
        )
    ):
        return _with_reason(script_decision, "outside-training-scope")
    if not 3 <= len(normalized.normalized_name) <= 19:
        return _with_reason(script_decision, "outside-training-scope")
    return script_decision


def normalize_for_lookup(
    name: NameInput, supported_scripts: frozenset[str]
) -> NormalizedFirstName:
    """Normalize one name for an exact table lookup without transliteration."""
    normalized = _normalize_common(name)
    if normalized.abstention_reason == "missing-name":
        return normalized
    script_decision = _with_script_decision(normalized, supported_scripts)
    if normalized.abstention_reason is not None:
        return NormalizedFirstName(
            normalized.name,
            normalized.normalized_name,
            script_decision.detected_script,
            script_decision.script_supported,
            normalized.abstention_reason,
        )
    if script_decision.abstention_reason is not None:
        return script_decision

    if normalized.normalized_name is None:
        raise RuntimeError(
            "eligible normalization unexpectedly produced a missing name"
        )
    if (
        not normalized.normalized_name.isascii()
        or not normalized.normalized_name.isalpha()
    ):
        return _with_reason(script_decision, "unsupported-characters")
    return script_decision


def _normalize_common(name: NameInput) -> NormalizedFirstName:
    if name is None:
        return NormalizedFirstName(None, None, None, None, "missing-name")

    normalized_name = unicodedata.normalize("NFC", name).strip().casefold()
    if not normalized_name:
        return NormalizedFirstName(name, normalized_name, None, None, "missing-name")
    if any(character.isspace() for character in normalized_name):
        return NormalizedFirstName(
            name,
            normalized_name,
            _detect_script(normalized_name),
            None,
            "not-single-first-name",
        )
    return NormalizedFirstName(
        name, normalized_name, _detect_script(normalized_name), None, None
    )


def _with_script_decision(
    normalized: NormalizedFirstName, supported_scripts: frozenset[str]
) -> NormalizedFirstName:
    detected_script = normalized.detected_script
    if detected_script is None:
        return _with_reason(normalized, "unsupported-characters")
    script_supported = detected_script in supported_scripts
    if not script_supported:
        return NormalizedFirstName(
            normalized.name,
            normalized.normalized_name,
            detected_script,
            False,
            "unsupported-script",
        )
    return NormalizedFirstName(
        normalized.name,
        normalized.normalized_name,
        detected_script,
        True,
        None,
    )


def _with_reason(normalized: NormalizedFirstName, reason: str) -> NormalizedFirstName:
    return NormalizedFirstName(
        normalized.name,
        normalized.normalized_name,
        normalized.detected_script,
        normalized.script_supported,
        reason,
    )


def _detect_script(value: str) -> str | None:
    scripts: set[str] = set()
    for character in value:
        if not character.isalpha():
            continue
        unicode_name = unicodedata.name(character, "")
        for marker, script in _UNICODE_SCRIPT_MARKERS.items():
            if marker in unicode_name:
                scripts.add(script)
                break
        else:
            scripts.add("Zzzz")
    if not scripts:
        return None
    if len(scripts) > 1:
        return "mixed"
    return next(iter(scripts))


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and math.isnan(value)
