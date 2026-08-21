# Copyright 2026 Timo Heimonen
# SPDX-License-Identifier: GPL-3.0-or-later

def _standard_result_accept($schema_version; $metric_layout):
  {
    accepted: true,
    metric_layout: $metric_layout,
    schema_version: $schema_version,
    diagnostic: null
  };

def _standard_result_reject($schema_version; $metric_layout; $diagnostic):
  {
    accepted: false,
    metric_layout: $metric_layout,
    schema_version: $schema_version,
    diagnostic: $diagnostic
  };

def _standard_result_normalize_integral_float:
  floor
  | tostring
  | capture(
      "^(?<sign>-?)(?<whole>[0-9]+)(?:\\.(?<fraction>[0-9]+))?(?:[eE](?<exponent>[+-]?[0-9]+))?$"
    ) as $parts
  | ($parts.fraction // "") as $fraction
  | ($parts.whole + $fraction) as $digits
  | (($parts.whole | length) + (($parts.exponent // "0") | tonumber)) as $decimal_position
  | (
      if $decimal_position <= 0 then
        "0"
      elif $decimal_position < ($digits | length) then
        $digits[0:$decimal_position]
      else
        $digits + ("0" * ($decimal_position - ($digits | length)))
      end
    ) as $integer_digits
  | (
      $integer_digits
      | sub("^0+"; "")
      | if . == "" then "0" else . end
    ) as $magnitude
  | (
      if $parts.sign == "-" and $magnitude != "0" then
        "-" + $magnitude
      else
        $magnitude
      end
    )
  | tonumber;

def _standard_result_completion:
  3 as $schema_version
  | "standard schema 3" as $prefix
  | if .status != "complete" then
      _standard_result_reject($schema_version; "headline_locality"; "\($prefix) field 'status' must equal 'complete'")
    elif (has("results_complete") | not) then
      _standard_result_reject($schema_version; "headline_locality"; "\($prefix) field 'results_complete' is required")
    elif (.results_complete | type) != "boolean" then
      _standard_result_reject(
        $schema_version;
        "headline_locality";
        "\($prefix) field 'results_complete' must be a boolean"
      )
    elif .results_complete != true then
      _standard_result_reject($schema_version; "headline_locality"; "\($prefix) field 'results_complete' must be true")
    elif (has("conclusions_valid") | not) then
      _standard_result_reject($schema_version; "headline_locality"; "\($prefix) field 'conclusions_valid' is required")
    elif (.conclusions_valid | type) != "boolean" then
      _standard_result_reject(
        $schema_version;
        "headline_locality";
        "\($prefix) field 'conclusions_valid' must be a boolean"
      )
    elif .conclusions_valid != true then
      _standard_result_reject($schema_version; "headline_locality"; "\($prefix) field 'conclusions_valid' must be true")
    else
      null
    end;

def standard_result_contract:
  if type != "object" then
    _standard_result_reject(null; null; "standard benchmark result must be a JSON object")
  elif .mode == "gpu_bandwidth"
      and (.schema_version | type) == "number"
      and .schema_version == 1 then
    _standard_result_reject(
      null;
      null;
      "GPU bandwidth schema 1 is not supported by this standard CPU result consumer"
    )
  elif (has("configuration") | not) then
    _standard_result_reject(null; null; "standard benchmark field 'configuration' is required")
  elif (.configuration | type) != "object" then
    _standard_result_reject(null; null; "standard benchmark field 'configuration' must be a JSON object")
  elif has("mode")
      or has("schema_version")
      or (.configuration | has("pattern_schema_version"))
      or (.configuration | has("schema_version"))
      or (.configuration | has("sweep_schema_version")) then
    _standard_result_reject(
      null;
      null;
      "standard benchmark result contains a conflicting mode/schema identity"
    )
  elif (.configuration | has("mode")) and .configuration.mode != "benchmark" then
    _standard_result_reject(
      null;
      null;
      "standard benchmark field 'configuration.mode' must equal 'benchmark'"
    )
  elif (.configuration | has("benchmark_schema_version") | not) then
    _standard_result_reject(
      null;
      null;
      "standard benchmark field 'configuration.benchmark_schema_version' is required"
    )
  else
    .configuration.benchmark_schema_version as $schema_version
    | ($schema_version | tostring) as $schema_text
    | if ($schema_version | type) != "number" then
        _standard_result_reject(null; null; "standard benchmark schema version must be integer 3")
      elif (($schema_text | test("^-?(0|[1-9][0-9]*)$")) | not)
          and ($schema_version | isinfinite) then
        _standard_result_reject(null; null; "standard benchmark schema version must be integer 3")
      elif (($schema_text | test("^-?(0|[1-9][0-9]*)$")) | not)
          and (($schema_version | floor) != $schema_version) then
        _standard_result_reject(null; null; "standard benchmark schema version must be integer 3")
      elif $schema_version != 3 then
        (
          if ($schema_text | test("^-?(0|[1-9][0-9]*)$")) then
            if $schema_version == 0 then 0 else $schema_version end
          else
            ($schema_version | _standard_result_normalize_integral_float)
          end
        ) as $normalized_schema_version
        | _standard_result_reject(
          $normalized_schema_version;
          null;
          "unsupported standard benchmark schema version: \($normalized_schema_version)"
        )
      elif (.configuration | has("mode") | not) then
        _standard_result_reject(
          $schema_version;
          null;
          "standard benchmark field 'configuration.mode' must equal 'benchmark'"
        )
      else
        _standard_result_completion as $completion_error
        | if $completion_error != null then
            $completion_error
          elif .configuration | has("output_file") then
            if (.configuration.output_file | type) != "string" then
              _standard_result_reject(
                $schema_version;
                "headline_locality";
                "standard schema 3 field 'configuration.output_file' must be a string"
              )
            else
              _standard_result_accept($schema_version; "headline_locality")
            end
          else
            _standard_result_reject(
              $schema_version;
              "headline_locality";
              "standard schema 3 field 'configuration.output_file' is required"
            )
          end
      end
  end;

def require_standard_result_contract:
  standard_result_contract
  | if .accepted then . else error(.diagnostic) end;
