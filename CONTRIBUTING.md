# Contributing to SVM Analyst

## Setting up your development environment

### 1. Clone the repository

```sh
git clone https://github.com/aminekhettat/SVM-analyst.git
cd SVM-analyst
```

### 2. Create a virtual environment and install dependencies

```sh
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

python -m pip install -r requirements.txt
```

### 3. Open in VS Code

```sh
code .
```

When VS Code opens the workspace it will prompt you to install the recommended
extensions listed in `.vscode/extensions.json`. Accept the prompt (or open the
**Extensions** view, filter by *@recommended*, and install them manually).

VS Code will also auto-detect the virtual environment created above through the
`python.defaultInterpreterPath` setting in `.vscode/settings.json`.

---

## Authenticating VS Code with GitHub (fine-grained token)

To push commits, manage pull requests, and use GitHub Copilot directly from
VS Code you need to authenticate. The recommended approach for this repository
is a **fine-grained personal access token (PAT)**.

### Step 1 – Generate the token on GitHub

1. Sign in to [github.com](https://github.com) with your account.
2. Navigate to **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens**
   (direct URL: <https://github.com/settings/tokens?type=beta>).
3. Click **Generate new token**.
4. Fill in the form:
   | Field | Recommended value |
   |---|---|
   | **Token name** | `SVM-analyst – VS Code` |
   | **Expiration** | 90 days (or set a custom date) |
   | **Resource owner** | `aminekhettat` |
   | **Repository access** | *Only select repositories* → `SVM-analyst` |
5. Under **Repository permissions**, grant at minimum:

   | Permission | Access level |
   |---|---|
   | Contents | Read and write |
   | Metadata | Read-only (mandatory) |
   | Pull requests | Read and write |
   | Issues | Read and write |
   | Workflows | Read and write |
   | Commit statuses | Read-only |

6. Click **Generate token** and **copy the token immediately** — it is shown
   only once.

### Step 2 – Configure Git to use the token

Option A – store it in the Git credential manager (recommended):

```sh
git config --global credential.helper store   # or "manager" on Windows
```

The first time you push, Git will ask for your credentials. Enter:

- **Username**: your GitHub username (`aminekhettat`)
- **Password**: the fine-grained token you just copied

Option B – embed it in the remote URL (less secure, avoid on shared machines):

```sh
git remote set-url origin https://<TOKEN>@github.com/aminekhettat/SVM-analyst.git
```

### Step 3 – Authenticate the GitHub Pull Requests extension in VS Code

1. Open VS Code.
2. Press **Ctrl+Shift+P** (or **Cmd+Shift+P** on macOS) and run
   **GitHub Pull Requests: Sign In to GitHub**.
3. When prompted, paste the fine-grained token.

You can now create/review pull requests, manage issues, and push commits
entirely from within VS Code.

---

## Running the tests

```sh
pytest
```

Or use the **Testing** panel in VS Code (the beaker icon in the Activity Bar).

## Code style

This project uses **Black** (formatter) and **Ruff** (linter).

```sh
python -m black .
python -m ruff check .
```

Both are configured as format-on-save in `.vscode/settings.json`.
