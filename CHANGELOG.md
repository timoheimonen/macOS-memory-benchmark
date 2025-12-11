# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.41] - 2025-12-11

### Added
- L1 and L2 cache latency testing using pointer chasing methodology.
- Cache latency results displayed with buffer size information.
- Cache latency statistics (average, min, max) across multiple loops.

### Changed
- Updated result printing functions to include cache latency results.

## [0.4] - 2025-12-10

### Fixed
- **ABI compatibility fix**: Fixed register corruption issue where all assembly functions (`memory_copy_loop_asm`, `memory_read_loop_asm`, `memory_write_loop_asm`) were using callee-saved registers q8-q15 (v8-v15) without preserving them, corrupting the calling C++ function's variables. All functions now use only caller-saved registers (q0-q7 and q16-q31) per AAPCS64, ensuring ABI compliance without requiring stack operations (maintaining leaf function optimization).
  - `memory_copy_loop_asm`: Replaced q8-q15 with q16-q23 and q0-q7 (reusing q0-q7 for remaining pairs).
  - `memory_read_loop_asm`: Moved accumulators from v8-v11 to v0-v3, and replaced all q8-q15 usage with q4-q7 and q16-q31 (data loaded first, then accumulated to avoid overwriting accumulators).
  - `memory_write_loop_asm`: Replaced q8-q15 with q16-q23 and q0-q7 (reusing q0-q7 since all values are zeros).

### Changed
- Help output (`-h`/`--help`) now includes author name, email, license, and link to GitHub page.


## [0.39] - 2025-12-07

### Fixed
- **Critical bug fix in `memory_read_loop_asm` main loop**: Fixed double accumulation bug where registers q0-q3 were loaded twice (at offsets 0-96 and 448-480) and accumulated twice, causing incorrect XOR checksum calculation. Reordered code to accumulate q0-q7 immediately after first load, then safely reuse q0-q3 for final loads, ensuring each data block is accumulated exactly once.

### Changed
- Bandwidth timing now starts after worker threads are created and waiting, and stops after join; added a start gate so setup/QoS overhead is excluded from measurements.
- Latency warm-up now runs immediately before each latency test loop to keep cache state consistent with the measurement that follows.
- Warmups for read/write/copy now run immediately before their respective benchmarks each loop to align cache/TLB state with the measurement.


## [0.38] - 2025-12-07

### Fixed
- **Critical bug fix in `memory_read_loop_asm` cleanup section**: Fixed register conflict in 256B chunk handling where accumulator registers `v8-v11` were being overwritten by loads into `q8-q11`, causing 128 bytes of data to be excluded from the XOR checksum calculation. Changed to use temporary registers `q16-q19` instead, ensuring all data is properly accumulated.

### Changed
- Standardized comments in `loops.s`: unified style, consistent alignment, added missing documentation.

