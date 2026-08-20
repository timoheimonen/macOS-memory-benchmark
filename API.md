# Machine-Readable Command-Line API

This document defines the supported process-level integration contract for `memory_benchmark` 0.62.0. It describes how
software launches a benchmark, separates machine-readable output from the human transcript, and decides whether a JSON
result is safe to consume. The generated Doxygen pages document C++ internals; they are not this process API.

Runtime behavior and executable integration tests are authoritative if this document and the implementation differ.

## Transport support in this revision

The first public slice enables the stdout JSON transport for direct standard and pattern commands. Other modes retain
their existing file-output contract until their mode-specific transport slices are implemented.

| Command | Real JSON file | Exact `--output -` stdout transport |
|---|---:|---:|
| Direct `--benchmark` | Yes | Yes |
| Direct `--patterns` | Yes | Yes |
| Standard or pattern `--sweep` | Yes | Not yet supported |
| Direct or sweep `--analyze-tlb` | Yes | Not yet supported |
| Direct or sweep `--analyze-core2core` | Yes | Not yet supported |
| Direct `--gpu-bandwidth` | Yes | Not yet supported |

Callers must use a real file path for every row whose stdout transport is not yet supported. GPU schema 1 does not
support sweeps.

## Invocation and stream contract

For a direct standard or pattern command, an output value that is exactly `-` selects stdout JSON:

```bash
memory_benchmark --benchmark --only-bandwidth --count 5 --buffer-size 512 --output -
```

The sentinel is classified from the raw option value before path normalization:

- `--output -` selects stdout JSON;
- `--output ./-` writes an ordinary file named `-` in the current directory;
- every other non-empty value is a file target;
- an omitted output option disables JSON output.

After successful parsing and mode selection, a supported machine-output command follows these stream rules:

- stdout contains exactly one complete JSON document followed by one newline;
- banners, configuration, progress, result tables, information, warnings, and runtime errors are written to stderr;
- no `-` or `-.tmp` transport file is created by the exact sentinel;
- the JSON payload is the mode's existing file-output payload, not a transport-specific wrapper;
- clients consume object keys by name and do not rely on textual key order.

Argument parsing and preflight validation can fail before a result state exists. Those failures return a non-zero process
status, write the centralized diagnostic to stderr, and leave stdout empty. Help is deliberately human-facing: `--help`
prints normal help to stdout and does not promise JSON even when combined with `--output -`.

Abrupt process termination, a crash, `SIGKILL`, or an unusable stdout pipe cannot guarantee a final document. Version 1
does not install a process-wide `SIGPIPE` policy.

## Checkpoints and final snapshots

Real file targets retain their existing persistence behavior:

- standard commands atomically checkpoint after completed loop-state changes and write their normal terminal result;
- pattern commands write their terminal payload through the shared atomic file writer;
- a temporary `<target>.tmp` file is replaced atomically, and a failed replacement preserves the preceding destination
  when possible.

Stdout is final-only. Intermediate standard checkpoint requests are successful no-ops, while all logical state changes,
stop observations, counters, cleanup, and final result construction still occur. The command serializes one terminal
snapshot after orchestration finishes. Stdout is not JSON Lines and never contains a sequence of checkpoint documents.

Use a real file target when crash-resilient intermediate standard state is required.

## Result schemas and completion

The stdout transport reuses the current mode payload and does not add a transport-version field. Process transport
contract version 1 first appears in software version `0.62.0`.

| Payload | Schema authority | A command result is complete only when |
|---|---|---|
| Standard | `configuration.benchmark_schema_version == 2` | `status == "complete" && results_complete == true` |
| Patterns | `configuration.pattern_schema_version == 3` | `status == "complete" && results_complete == true` |

Command completeness does not make every optional metric available. A selected standard measurement must have its
mode-specific measured/quality state and a non-null value. A pattern measurement may be intentionally `skipped` while
the command remains complete; consumers of a particular pattern metric must require `status == "measured"` and a
non-null value.

Graceful interruption or runtime failure after a representable result state has been initialized emits the available
partial or failed JSON snapshot. The execution status and payload are independent: a non-zero status must not cause a
caller to discard evidence without parsing it, and exit status zero does not prove that conclusions are complete.

## Consumer acceptance procedure

A caller accepts a standard or pattern conclusion only after all of the following checks succeed:

1. Launch the executable with an argv array; do not construct an unquoted shell command from external input.
2. Capture stdout and stderr separately and wait for the process and both streams to finish.
3. Parse stdout as exactly one JSON document with no trailing non-whitespace data.
4. Check the process result for transport or runtime failure.
5. Check the supported mode and schema-version field.
6. Apply the mode-specific command-completeness predicate above.
7. Apply the selected metric's status, non-null, and quality predicates.

Language-neutral pseudocode:

```text
result = run_process(argv, capture_stdout=true, capture_stderr=true)
if result.stdout is empty:
    reject_with_diagnostic(result.exit_status, result.stderr)

document = parse_one_json_value(result.stdout)
require_only_whitespace_after_document()
require_supported_mode_and_schema(document)
require_process_completed_as_expected(result.exit_status)
require_mode_completion_predicate(document)
require_selected_metric_is_measured_and_non_null(document)
accept(document)
```

Shell capture example:

```bash
memory_benchmark --patterns --buffer-size 512 --count 5 --seed 42 --output - \
  >patterns.json 2>patterns.log
jq -e '.configuration.pattern_schema_version == 3 and
       .status == "complete" and .results_complete == true' patterns.json
```

## Compatibility policy

- `version` identifies the application release; it is not a result schema version.
- Standard schema 2 and pattern schema 3 remain authoritative at their existing locations.
- Additive optional fields may remain within a schema version only when old consumers can safely ignore them.
- Removing or renaming a field, changing its type, or changing its meaning requires a mode schema-version bump.
- A methodology change that affects comparison requires the mode's methodology-version mechanism even when JSON shape
  is unchanged.
- A transport change alone does not change the measurement schema.
- Schema-location normalization belongs in client code; this API does not move existing version fields.

## Benchmark process policy

Benchmark commands should be serialized and run on an otherwise idle machine. Avoid overlapping benchmark processes,
keep power and thermal conditions controlled, and use `caffeinate -i -d` for long runs when sleep would invalidate the
experiment. Process isolation is intentional: signals, QoS, large mappings, ARM64 work, and Metal resource lifetime stay
inside the launched benchmark process.
