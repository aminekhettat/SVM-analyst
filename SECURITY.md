# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.1.x   | Yes       |
| < 1.1   | No        |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub Issues.**

If you discover a security vulnerability in SVM Analyst, please report it privately so it can be assessed and patched before public disclosure.

**How to report:**

1. Go to the [Security tab](https://github.com/aminekhettat/SVM-analyst/security) of the repository.
2. Click **"Report a vulnerability"** to open a private advisory.
3. Describe the vulnerability, the version affected, and steps to reproduce.

You will receive an acknowledgement within **72 hours**. After triage, a fix will be prepared and a patched release published. You will be credited in the release notes if you wish.

## Scope

SVM Analyst is a local desktop application. The main security considerations are:

- **Malicious config files (JSON)**: The application loads JSON configuration files. Do not load config files from untrusted sources.
- **Exported PDF/CSV**: Generated output files are based on simulation data only and contain no user credentials.
- **Dependencies**: Third-party libraries (PySide6, numpy, scipy, matplotlib, pyqtgraph) may have their own vulnerabilities. Keep dependencies up to date.

## Out of scope

- Vulnerabilities in Python itself or in third-party packages not specific to this project.
- Theoretical attacks with no plausible real-world impact on this tool.
