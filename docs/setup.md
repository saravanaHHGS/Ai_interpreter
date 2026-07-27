# Setup

From a completely fresh Windows machine to a working development environment.
Nothing is assumed to be installed. Every command is explained.

---

## 1. Install Python 3.12

Download from <https://www.python.org/downloads/windows/>. Choose 3.12 or
newer, 64-bit.

**During installation, tick "Add python.exe to PATH".** Without it, the `py`
launcher may still work but `python` will not, and half the tutorials you find
online will appear broken.

Verify:

```powershell
py -0p
```

Expected: a line containing `3.12` and a path. If nothing is listed, Python is
not installed correctly.

**Why 3.12 specifically?** The code uses `StrEnum`, `Self`, and PEP 695
generics-adjacent syntax that require 3.11+, and 3.12 is the version every
dependency in the project ships a prebuilt Windows wheel for.

---

## 2. Install Git

Download from <https://git-scm.com/download/win>. Accept the defaults.

```powershell
git --version
```

---

## 3. Get the project

```powershell
git clone <repository-url> C:\ai_interpreter
cd C:\ai_interpreter
```

---

## 4. Run the bootstrap script

```powershell
.\scripts\bootstrap.ps1
```

If PowerShell refuses with a message about execution policies, allow local
scripts for your user only:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

This is the standard setting for a development machine: it permits scripts you
wrote locally while still requiring a signature on scripts downloaded from the
internet.

### What the script does, step by step

| Step | Command it runs | Why |
|---|---|---|
| 1 | `py -0p` | Confirms Python 3.12+ exists before doing anything |
| 2 | `py -3.12 -m venv .venv` | Creates an isolated environment in `.venv` |
| 3 | `pip install setuptools wheel` | Provides the build backend |
| 4 | `pip install -r requirements-dev.txt` | Installs the project and dev tools |
| 5 | copies `.env.example` to `.env` | Creates the secrets file (empty is correct) |
| 6 | `python -m ai_interpreter --check` | Verifies the result |

**Why a virtual environment?** It keeps this project's dependencies separate
from every other Python project and from Windows' own Python. Without it, two
projects needing different numpy versions cannot coexist, and a bad install
can break unrelated software.

**Why `.venv` and not `venv`?** The leading dot is the convention most editors
and tools auto-detect, including VS Code.

**Why the script does not upgrade pip:** on Windows, pip upgrading itself
while running can fail with `WinError 32: file in use` and leave the
environment with a broken pip that cannot install anything. This actually
happened while building this project. The bundled pip works fine.

---

## 5. Verify

```powershell
.\run.ps1 --check
```

You should see a report covering hardware, the selected profile, paths,
packages, tools, audio endpoints and privacy settings, ending with
`All required checks passed.`

Two warnings are expected at this stage and are not problems:

- **ffmpeg is not on PATH** — needed from Phase 3.
- **No virtual audio cable detected** — needed from Phase 7.

---

## 6. Install FFmpeg (needed from Phase 3)

The easiest route on Windows 11:

```powershell
winget install --id Gyan.FFmpeg -e
```

Then **close and reopen PowerShell** so the updated PATH is picked up, and
confirm:

```powershell
ffmpeg -version
```

If `winget` is unavailable, download a build from
<https://www.gyan.dev/ffmpeg/builds/>, extract it, and add its `bin` folder to
your PATH manually.

**Why FFmpeg?** Converting recordings and test fixtures between formats and
sample rates. It is a command line tool, not a Python package.

---

## 7. Install VB-CABLE (needed from Phase 7 — do it early)

This one **requires a reboot**, so install it now rather than discovering it
later.

1. Download from <https://vb-audio.com/Cable/>.
2. Extract the zip.
3. Right-click `VBCABLE_Setup_x64.exe` → **Run as administrator**.
4. Click *Install Driver*.
5. **Reboot.**

Verify:

```powershell
.\run.ps1 --check
```

The audio endpoints section should now list `CABLE Input` and `CABLE Output`,
tagged `[CABLE]`.

**What it does:** creates a loopback pair. Anything played to *CABLE Input*
appears as microphone audio on *CABLE Output*. Teams selects CABLE Output as
its microphone and hears the translated speech.

**Licence note:** VB-CABLE is donationware and **may not be redistributed**.
The Phase 12 installer will link to it rather than bundle it.

---

## 8. Configure the application

### Normal settings — YAML

Edit `config/default.yaml`. It is commented throughout. For personal changes
that should survive a `git pull`, create:

```
%APPDATA%\HikeHealthGS\ai-interpreter\config.yaml
```

and put only the keys you want to change in it:

```yaml
vad:
  min_silence_ms: 450
```

### One-off overrides — environment variables

```powershell
$env:AI_INTERPRETER__STT__MODEL = "base"
.\run.ps1 --print-config
```

Format: `AI_INTERPRETER__SECTION__KEY`. Double underscores separate levels.

### Secrets — `.env`

`.env` holds credentials only, and is git-ignored. Both values may stay empty:
the application is local-first and needs neither for normal use.

```
AI_INTERPRETER_HF_TOKEN=
AI_INTERPRETER_NVIDIA_NIM_API_KEY=
```

**Never put a token in a YAML file.** The schema has no field for one and
would reject it, but the real reason is that YAML files are committed to git
and `.env` is not.

---

## 9. Editor setup (VS Code)

Install the **Python** extension, then select the interpreter:

`Ctrl+Shift+P` → *Python: Select Interpreter* → `.\.venv\Scripts\python.exe`

Optionally install the **Ruff** extension for inline lint and format-on-save.

---

## 10. Everyday commands

```powershell
.\run.ps1 --check                  # environment doctor
.\run.ps1 --print-config           # effective settings
.\scripts\quality.ps1              # lint, types, tests
.\scripts\quality.ps1 -Fix         # auto-fix formatting and safe lint issues
.\scripts\bootstrap.ps1 -Recreate  # rebuild a broken environment from scratch
```

---

## 11. Common setup mistakes

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: ai_interpreter` | Using system Python, not the venv | Use `.\run.ps1`, or select the venv interpreter in VS Code |
| `running scripts is disabled on this system` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No module named 'pip._internal.cli'` | An interrupted pip self-upgrade | `.\scripts\bootstrap.ps1 -Recreate` |
| `--check` exits 1 with a missing package | Dependencies not installed | `.\scripts\bootstrap.ps1` |
| Configuration change has no effect | Edited the wrong layer | `.\run.ps1 --print-config` lists every source in precedence order |
| `UnicodeEncodeError` printing Tamil | Legacy console code page | Already handled; report it if you see it |

More in [troubleshooting.md](troubleshooting.md).
