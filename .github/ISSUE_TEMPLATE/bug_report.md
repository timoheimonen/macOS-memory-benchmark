---
name: Bug Report
about: Report a bug or unexpected behavior in macOS-memory-benchmark
title: "[BUG] "
labels: bug
assignees: ''
---

## Issue Description
A clear and concise description of the issue.

## Environment
**System Information:**
- macOS Version: [e.g., macOS 26.2]
- Chip: [e.g., Apple M5, M4, M3, M2, M1]
- Performance Cores: [e.g., 4]
- Efficiency Cores: [e.g., 6]
- Total Memory: [e.g., 24GB]

**Benchmark Version:**
- Version: [Run `./memory_benchmark --version` or check version.h]
- Build Method: [e.g., make, Homebrew]

## Steps to Reproduce
1. Command or configuration used:
   ```bash
   ./memory_benchmark [your arguments here]
   ```
2. Any specific setup or conditions
3. Expected behavior vs. actual behavior

## Actual Behavior
What actually happened? Include:
- Error messages (full output)
- Unexpected results
- Performance issues
- Crashes or hangs

## Expected Behavior
What should have happened?

## Output/Logs
```
[Paste relevant output, error messages, or stack traces here]
```

## Additional Context
- Does it happen consistently or intermittently?
- Any background processes that might affect benchmarking?
- Thermal conditions (sustained load, throttling)?
- Any modifications to the codebase?

## Configuration Used
If applicable, share your command-line arguments or configuration:
```bash
./memory_benchmark -buffersize 1024 -iterations 2000 [etc]
```

## Possible Solution
If you have ideas about what might be causing the issue or how to fix it, please share.
