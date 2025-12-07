# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.39] - 2025-12-07

### Fixed
- **Critical bug fix in `memory_read_loop_asm` main loop**: Fixed double accumulation bug where registers q0-q3 were loaded twice (at offsets 0-96 and 448-480) and accumulated twice, causing incorrect XOR checksum calculation. Reordered code to accumulate q0-q7 immediately after first load, then safely reuse q0-q3 for final loads, ensuring each data block is accumulated exactly once.


## [0.38] - 2025-12-07

### Fixed
- **Critical bug fix in `memory_read_loop_asm` cleanup section**: Fixed register conflict in 256B chunk handling where accumulator registers `v8-v11` were being overwritten by loads into `q8-q11`, causing 128 bytes of data to be excluded from the XOR checksum calculation. Changed to use temporary registers `q16-q19` instead, ensuring all data is properly accumulated.

### Changed
- Standardized comments in `loops.s`: unified style, consistent alignment, added missing documentation.

