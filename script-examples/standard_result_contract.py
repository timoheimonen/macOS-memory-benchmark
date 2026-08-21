#!/usr/bin/env python3
# Copyright 2026 Timo Heimonen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Strict current-contract validation for standard benchmark JSON results."""

import math
from typing import Any, Mapping, NamedTuple, Optional


HEADLINE_LOCALITY_LAYOUT = "headline_locality"


class StandardResultValidation(NamedTuple):
    """Contract outcome and independently identified metric-layout family."""

    accepted: bool
    metric_layout: Optional[str]
    schema_version: Optional[int]
    diagnostic: Optional[str]


class StandardResultContractError(RuntimeError):
    """Raised when a standard result does not satisfy its versioned contract."""


def _accepted(schema_version: Optional[int], metric_layout: str) -> StandardResultValidation:
    return StandardResultValidation(True, metric_layout, schema_version, None)


def _rejected(
    schema_version: Optional[int],
    diagnostic: str,
    metric_layout: Optional[str] = None,
) -> StandardResultValidation:
    return StandardResultValidation(False, metric_layout, schema_version, diagnostic)


def _normalize_integral_float(value: float) -> int:
    """Canonicalize a finite integral JSON float through its shortest decimal form."""

    text = repr(value).lower()
    sign = -1 if text.startswith("-") else 1
    if text[:1] in ("+", "-"):
        text = text[1:]

    if "e" in text:
        mantissa, exponent_text = text.split("e", 1)
        exponent = int(exponent_text)
    else:
        mantissa = text
        exponent = 0

    if "." in mantissa:
        whole, fraction = mantissa.split(".", 1)
    else:
        whole = mantissa
        fraction = ""

    digits = whole + fraction
    decimal_position = len(whole) + exponent
    if decimal_position <= 0:
        integer_digits = "0"
    elif decimal_position < len(digits):
        integer_digits = digits[:decimal_position]
    else:
        integer_digits = digits + ("0" * (decimal_position - len(digits)))

    magnitude = integer_digits.lstrip("0") or "0"
    return 0 if magnitude == "0" else sign * int(magnitude)


def _validate_completion_fields(data: Mapping[str, Any]) -> Optional[StandardResultValidation]:
    schema_version = 3
    prefix = "standard schema 3"

    if data.get("status") != "complete":
        return _rejected(
            schema_version,
            f"{prefix} field 'status' must equal 'complete'",
            HEADLINE_LOCALITY_LAYOUT,
        )

    if "results_complete" not in data:
        return _rejected(
            schema_version,
            f"{prefix} field 'results_complete' is required",
            HEADLINE_LOCALITY_LAYOUT,
        )
    if not isinstance(data["results_complete"], bool):
        return _rejected(
            schema_version,
            f"{prefix} field 'results_complete' must be a boolean",
            HEADLINE_LOCALITY_LAYOUT,
        )
    if not data["results_complete"]:
        return _rejected(
            schema_version,
            f"{prefix} field 'results_complete' must be true",
            HEADLINE_LOCALITY_LAYOUT,
        )

    if "conclusions_valid" not in data:
        return _rejected(
            schema_version,
            f"{prefix} field 'conclusions_valid' is required",
            HEADLINE_LOCALITY_LAYOUT,
        )
    if not isinstance(data["conclusions_valid"], bool):
        return _rejected(
            schema_version,
            f"{prefix} field 'conclusions_valid' must be a boolean",
            HEADLINE_LOCALITY_LAYOUT,
        )
    if not data["conclusions_valid"]:
        return _rejected(
            schema_version,
            f"{prefix} field 'conclusions_valid' must be true",
            HEADLINE_LOCALITY_LAYOUT,
        )
    return None


def validate_standard_result(data: Any) -> StandardResultValidation:
    """Classify a complete current standard result without reading its metrics."""

    if not isinstance(data, Mapping):
        return _rejected(None, "standard benchmark result must be a JSON object")

    gpu_schema_version = data.get("schema_version")
    gpu_schema_is_one = (
        isinstance(gpu_schema_version, int)
        and not isinstance(gpu_schema_version, bool)
        and gpu_schema_version == 1
    ) or (
        isinstance(gpu_schema_version, float)
        and math.isfinite(gpu_schema_version)
        and gpu_schema_version == 1
    )
    if (
        data.get("mode") == "gpu_bandwidth"
        and gpu_schema_is_one
    ):
        return _rejected(
            None,
            "GPU bandwidth schema 1 is not supported by this standard CPU result consumer",
        )

    if "configuration" not in data:
        return _rejected(None, "standard benchmark field 'configuration' is required")

    configuration = data["configuration"]
    if not isinstance(configuration, Mapping):
        return _rejected(None, "standard benchmark field 'configuration' must be a JSON object")

    alternative_schema_keys = (
        "pattern_schema_version",
        "schema_version",
        "sweep_schema_version",
    )
    if (
        "mode" in data
        or "schema_version" in data
        or any(key in configuration for key in alternative_schema_keys)
    ):
        return _rejected(
            None,
            "standard benchmark result contains a conflicting mode/schema identity",
        )

    if "mode" in configuration and configuration["mode"] != "benchmark":
        return _rejected(
            None,
            "standard benchmark field 'configuration.mode' must equal 'benchmark'",
        )

    if "benchmark_schema_version" not in configuration:
        return _rejected(
            None,
            "standard benchmark field 'configuration.benchmark_schema_version' is required",
        )

    raw_schema_version = configuration["benchmark_schema_version"]
    if isinstance(raw_schema_version, bool):
        return _rejected(None, "standard benchmark schema version must be integer 3")
    if isinstance(raw_schema_version, int):
        schema_version = raw_schema_version
    elif (
        isinstance(raw_schema_version, float)
        and math.isfinite(raw_schema_version)
        and raw_schema_version.is_integer()
    ):
        schema_version = _normalize_integral_float(raw_schema_version)
    else:
        return _rejected(None, "standard benchmark schema version must be integer 3")
    if schema_version != 3:
        return _rejected(
            schema_version,
            f"unsupported standard benchmark schema version: {schema_version}",
        )

    if "mode" not in configuration:
        return _rejected(
            schema_version,
            "standard benchmark field 'configuration.mode' must equal 'benchmark'",
        )

    completion_error = _validate_completion_fields(data)
    if completion_error is not None:
        return completion_error

    if "output_file" in configuration and not isinstance(configuration["output_file"], str):
        return _rejected(
            schema_version,
            "standard schema 3 field 'configuration.output_file' must be a string",
            HEADLINE_LOCALITY_LAYOUT,
        )
    if "output_file" not in configuration:
        return _rejected(
            schema_version,
            "standard schema 3 field 'configuration.output_file' is required",
            HEADLINE_LOCALITY_LAYOUT,
        )

    return _accepted(schema_version, HEADLINE_LOCALITY_LAYOUT)


def require_standard_result(data: Any, source: Optional[str] = None) -> StandardResultValidation:
    """Return an accepted contract result or raise a stable contract diagnostic."""

    validation = validate_standard_result(data)
    if validation.accepted:
        return validation

    diagnostic = validation.diagnostic or "standard benchmark result was rejected"
    if source:
        diagnostic = f"{diagnostic}: {source}"
    raise StandardResultContractError(diagnostic)
