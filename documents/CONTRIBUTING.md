# Contributing

Thank you for your interest in contributing to this project.
Contributions, feedback, and discussions are welcome.

Because this tool focuses on low-level memory behavior and performance measurement, changes are reviewed carefully to ensure correctness and repeatability.

---

## Pull Requests

Please follow these guidelines when submitting a pull request:

### 1. Target branch

- Open pull requests against the `development` branch.
- The maintainer may redirect the PR to a temporary preview/review branch for detailed testing before integration.

You do not need to take any action if the base branch is changed by the maintainer.

---

### 2. Scope of changes

Keep pull requests focused and minimal:

- One logical change per PR is preferred.
- Avoid unrelated refactoring or formatting-only changes unless explicitly discussed.

---

### 3. Performance-critical code

Extra care is required for changes that affect:

- measurement loops
- timing logic
- memory allocation or access patterns
- inline assembly or compiler-sensitive code paths

If your PR modifies any of the above, please explain:

- what changed
- why it is correct
- how it was validated

---

### 4. Testing

Before submitting a PR, ensure that:

- the project builds successfully (e.g. `make`)
- basic benchmark runs complete without errors (e.g. `./memory_benchmark -h` and a short run)
- results are consistent with existing behavior (unless the change explicitly modifies semantics)

If you run tests:

- Install GoogleTest: `brew install googletest`
- Run unit tests: `make test`

If possible, include example output or notes about observed behavior.

---

### 5. Documentation

If your change affects:

- output format
- command-line options
- measurement semantics

please update the relevant documentation or mention why no documentation change is needed.

---

## Code Style

- Follow the existing coding style and structure.
- Avoid introducing new dependencies unless necessary.
- Prefer clarity and explicitness over clever optimizations.

---

## Discussion

If you are unsure whether a change is appropriate, feel free to:

- open an issue first, or
- start a draft pull request for discussion.

---

## License

By contributing, you agree that your contributions will be licensed under the same license as this project.

---

## Code of Conduct

By participating in this project you agree to abide by the Code of Conduct: `CODE_OF_CONDUCT.md`.

## Security

For security vulnerabilities, please follow the reporting guidance in `SECURITY.md`.
