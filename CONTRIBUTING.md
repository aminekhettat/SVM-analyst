# Contributing to SVM Analyst

Thank you for your interest in contributing! This document explains how to get started.

## Table of contents

1. [Reporting bugs](#reporting-bugs)
2. [Requesting features](#requesting-features)
3. [Development setup](#development-setup)
4. [Running tests](#running-tests)
5. [Code style](#code-style)
6. [Submitting a pull request](#submitting-a-pull-request)
7. [Branch naming convention](#branch-naming-convention)

---

## Reporting bugs

Use the **Bug Report** issue template on GitHub Issues. Include:

- SVM Analyst version (shown in window title or `CHANGELOG.txt`)
- Operating system and Python version (if running from source)
- Exact steps to reproduce the issue
- The full error message or traceback

For private/security issues, see [SECURITY.md](SECURITY.md).

## Requesting features

Use the **Feature Request** issue template. Describe the problem you are trying to solve and your proposed solution. Screenshots or references to academic papers are welcome.

## Development setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-fork>/SVM-analyst.git
cd "SVM-analyst"

# 2. Create a virtual environment (Python 3.12 recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install all dependencies
pip install -r requirements.txt
```

## Running tests

```bash
# All 191 tests (includes slow sweep/simulation tests ~90 s)
pytest tests/ -q

# Fast subset only (excludes the slow sweep test)
pytest tests/ -q --ignore=tests/test_sweep.py

# With environment variables for headless Qt
$env:MPLBACKEND='Agg'; $env:QT_QPA_PLATFORM='offscreen'
pytest tests/ -q
```

All tests must pass before submitting a pull request.

## Code style

This project uses **black** (formatter) and **ruff** (linter).

```bash
# Format
black svm_shaper/ tests/ scripts/

# Lint
ruff check svm_shaper/ tests/ scripts/
```

Zero ruff warnings and zero diagnostics are required before every commit.

## Submitting a pull request

1. Create a branch from `main` using the naming convention below.
2. Make your changes, add or update tests to cover all new code surfaces.
3. Run the full test suite and confirm 0 failures.
4. Run black and ruff; fix all warnings.
5. Add a `CHANGELOG.txt` entry describing your change.
6. Push and open a pull request against `main`. Fill in the PR template.

Pull requests that do not have 100% test coverage of new code, or that introduce lint warnings, will not be merged.

## Branch naming convention

| Type          | Pattern                     | Example                         |
| ------------- | --------------------------- | ------------------------------- |
| Bug fix       | `fix/<short-description>`   | `fix/scipy-missing-from-bundle` |
| Feature       | `feat/<short-description>`  | `feat/export-to-pdf`            |
| Documentation | `docs/<short-description>`  | `docs/update-user-manual`       |
| Chore / CI    | `chore/<short-description>` | `chore/update-pyinstaller`      |

---

By contributing you agree that your contributions will be licensed under the same license as the project. See [LICENSE](LICENSE) for details.
