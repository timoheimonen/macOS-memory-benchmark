# Copyright 2026 Timo Heimonen
# SPDX-License-Identifier: GPL-3.0-or-later

def _standard_result_accept($schema_version; $metric_layout):
  {
    accepted: true,
    metric_layout: $metric_layout,
    schema_version: $schema_version,
    diagnostic: null
  };

def _standard_result_reject($schema_version; $diagnostic):
  {
    accepted: false,
    metric_layout: (
      if $schema_version == 2 or $schema_version == 3 then "headline_locality" else null end
    ),
    schema_version: $schema_version,
    diagnostic: $diagnostic
  };

def _standard_result_completion($schema_version):
  "standard schema \($schema_version)" as $prefix
  | if .status != "complete" then
      _standard_result_reject($schema_version; "\($prefix) field 'status' must equal 'complete'")
    elif (has("results_complete") | not) then
      _standard_result_reject($schema_version; "\($prefix) field 'results_complete' is required")
    elif (.results_complete | type) != "boolean" then
      _standard_result_reject($schema_version; "\($prefix) field 'results_complete' must be a boolean")
    elif .results_complete != true then
      _standard_result_reject($schema_version; "\($prefix) field 'results_complete' must be true")
    elif $schema_version == 2 and (has("conclusions_valid") | not) then
      null
    elif (has("conclusions_valid") | not) then
      _standard_result_reject($schema_version; "\($prefix) field 'conclusions_valid' is required")
    elif (.conclusions_valid | type) != "boolean" then
      if $schema_version == 2 then
        _standard_result_reject(
          $schema_version;
          "\($prefix) field 'conclusions_valid' must be a boolean when present"
        )
      else
        _standard_result_reject($schema_version; "\($prefix) field 'conclusions_valid' must be a boolean")
      end
    elif .conclusions_valid != true then
      if $schema_version == 2 then
        _standard_result_reject(
          $schema_version;
          "\($prefix) field 'conclusions_valid' must be true when present"
        )
      else
        _standard_result_reject($schema_version; "\($prefix) field 'conclusions_valid' must be true")
      end
    else
      null
    end;

def standard_result_contract:
  if type != "object" then
    _standard_result_reject(null; "standard benchmark result must be a JSON object")
  elif .mode == "gpu_bandwidth" and .schema_version == 1 then
    _standard_result_reject(
      null;
      "GPU bandwidth schema 1 is not supported by this standard CPU result consumer"
    )
  elif (has("configuration") | not) then
    _standard_result_accept(null; "historical_average_tlb")
  elif (.configuration | type) != "object" then
    _standard_result_reject(null; "standard benchmark field 'configuration' must be a JSON object")
  elif (.configuration | has("benchmark_schema_version") | not) then
    _standard_result_accept(null; "historical_average_tlb")
  else
    .configuration.benchmark_schema_version as $schema_version
    | if ($schema_version | type) != "number" or ($schema_version | floor) != $schema_version then
        _standard_result_reject(null; "standard benchmark schema version must be integer 2 or 3")
      elif $schema_version != 2 and $schema_version != 3 then
        _standard_result_reject(
          $schema_version;
          "unsupported standard benchmark schema version: \($schema_version)"
        )
      else
        _standard_result_completion($schema_version) as $completion_error
        | if $completion_error != null then
            $completion_error
          elif .configuration | has("output_file") then
            if (.configuration.output_file | type) != "string" then
              if $schema_version == 2 then
                _standard_result_reject(
                  $schema_version;
                  "standard schema 2 field 'configuration.output_file' must be a string when present"
                )
              else
                _standard_result_reject(
                  $schema_version;
                  "standard schema 3 field 'configuration.output_file' must be a string"
                )
              end
            else
              _standard_result_accept($schema_version; "headline_locality")
            end
          elif $schema_version == 3 then
            _standard_result_reject(
              $schema_version;
              "standard schema 3 field 'configuration.output_file' is required"
            )
          else
            _standard_result_accept($schema_version; "headline_locality")
          end
      end
  end;

def require_standard_result_contract:
  standard_result_contract
  | if .accepted then . else error(.diagnostic) end;
