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

"""Version-aware contract validation for standard benchmark JSON results."""

import math
from typing import Any, Mapping, NamedTuple, Optional


HEADLINE_LOCALITY_LAYOUT = "headline_locality"
HISTORICAL_AVERAGE_TLB_LAYOUT = "historical_average_tlb"


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


def _rejected(schema_version: Optional[int], diagnostic: str) -> StandardResultValidation:
    metric_layout = HEADLINE_LOCALITY_LAYOUT if schema_version in (2, 3) else None
    return StandardResultValidation(False, metric_layout, schema_version, diagnostic)


def _validate_completion_fields(
    data: Mapping[str, Any], schema_version: int
) -> Optional[StandardResultValidation]:
    prefix = f"standard schema {schema_version}"

    if data.get("status") != "complete":
        return _rejected(schema_version, f"{prefix} field 'status' must equal 'complete'")

    if "results_complete" not in data:
        return _rejected(schema_version, f"{prefix} field 'results_complete' is required")
    if not isinstance(data["results_complete"], bool):
        return _rejected(schema_version, f"{prefix} field 'results_complete' must be a boolean")
    if not data["results_complete"]:
        return _rejected(schema_version, f"{prefix} field 'results_complete' must be true")

    if schema_version == 2:
        if "conclusions_valid" not in data:
            return None
        if not isinstance(data["conclusions_valid"], bool):
            return _rejected(
                schema_version,
                f"{prefix} field 'conclusions_valid' must be a boolean when present",
            )
        if not data["conclusions_valid"]:
            return _rejected(
                schema_version,
                f"{prefix} field 'conclusions_valid' must be true when present",
            )
        return None

    if "conclusions_valid" not in data:
        return _rejected(schema_version, f"{prefix} field 'conclusions_valid' is required")
    if not isinstance(data["conclusions_valid"], bool):
        return _rejected(schema_version, f"{prefix} field 'conclusions_valid' must be a boolean")
    if not data["conclusions_valid"]:
        return _rejected(schema_version, f"{prefix} field 'conclusions_valid' must be true")
    return None


def validate_standard_result(data: Any) -> StandardResultValidation:
    """Classify a standard result without reading metrics or raising an exception.

    Released schema 2 and current schema 3 share the headline/locality metric
    layout but have different completion contracts. A result with no standard
    schema identity is routed to the explicitly supported historical metric
    layout. Any explicit, unsupported identity is rejected.
    """

    if not isinstance(data, Mapping):
        return _rejected(None, "standard benchmark result must be a JSON object")

    if data.get("mode") == "gpu_bandwidth" and data.get("schema_version") == 1:
        return _rejected(
            None,
            "GPU bandwidth schema 1 is not supported by this standard CPU result consumer",
        )

    if "configuration" not in data:
        return _accepted(None, HISTORICAL_AVERAGE_TLB_LAYOUT)

    configuration = data["configuration"]
    if not isinstance(configuration, Mapping):
        return _rejected(None, "standard benchmark field 'configuration' must be a JSON object")

    if "benchmark_schema_version" not in configuration:
        return _accepted(None, HISTORICAL_AVERAGE_TLB_LAYOUT)

    raw_schema_version = configuration["benchmark_schema_version"]
    if isinstance(raw_schema_version, bool):
        return _rejected(None, "standard benchmark schema version must be integer 2 or 3")
    if isinstance(raw_schema_version, int):
        schema_version = raw_schema_version
    elif (
        isinstance(raw_schema_version, float)
        and math.isfinite(raw_schema_version)
        and raw_schema_version.is_integer()
    ):
        schema_version = int(raw_schema_version)
    else:
        return _rejected(None, "standard benchmark schema version must be integer 2 or 3")
    if schema_version not in (2, 3):
        return _rejected(
            schema_version,
            f"unsupported standard benchmark schema version: {schema_version}",
        )

    completion_error = _validate_completion_fields(data, schema_version)
    if completion_error is not None:
        return completion_error

    if "output_file" in configuration and not isinstance(configuration["output_file"], str):
        suffix = "when present" if schema_version == 2 else ""
        diagnostic = f"standard schema {schema_version} field 'configuration.output_file' must be a string"
        if suffix:
            diagnostic += f" {suffix}"
        return _rejected(schema_version, diagnostic)
    if schema_version == 3 and "output_file" not in configuration:
        return _rejected(
            schema_version,
            "standard schema 3 field 'configuration.output_file' is required",
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
