# Deployment

How AI Interpreter is packaged, installed and licensed. Revised in Phase 12
to match what the project *learned* between Phases 1 and 11 - most
importantly about Smart App Control.

---

## 1. Why the distribution is a source archive, not an installer EXE

Phase 1 planned PyInstaller + Inno Setup. Phase 12 rejects both, on
evidence gathered on the reference machine:

**Smart App Control blocks unsigned binaries.** It blocked PyTorch's DLLs
(permanently), mypy's compiled runtime (forcing a from-source install), and
it decides per-file with no user override short of disabling the policy -
a one-way door this project refuses to ask users to walk through. A
PyInstaller bootloader without a code-signing certificate is exactly the
kind of unsigned, low-reputation executable SAC exists to stop. Shipping
one would produce an installer that fails precisely on the well-managed
Windows 11 machines this project targets.

A **source distribution** has no such problem: every binary that actually
executes is signed by someone SAC trusts - `python.exe` from python.org,
wheels from PyPI (including Qt, signed by the Qt Company - verified to load
under enforced SAC in Phase 8). The application itself is pure Python.

Revisit only if a code-signing certificate is acquired; the release
checklist below already has the slot.

## 2. Building a release

```powershell
.\scripts\package.ps1        # runs the fast suite, then builds
                             # dist\ai-interpreter-<version>.zip
```

The archive contains the source tree, configuration, docs, scripts and
pinned requirements - no virtual environment, no models, no recordings, no
`.env`, no git history. It is a few hundred kilobytes.

## 3. Installing on a new machine

1. Install **Python 3.12+** from <https://www.python.org/downloads/>,
   ticking "Add python.exe to PATH".
2. Unzip the archive, e.g. to `C:\ai_interpreter`.
3. `.\scripts\bootstrap.ps1 -Locked` - creates the environment from the
   exact pinned versions the release was tested with, then runs the
   environment doctor (`--check`).
4. Install **VB-CABLE** yourself from <https://vb-audio.com/Cable/>
   (download, extract, right-click `VBCABLE_Setup_x64.exe` -> Run as
   administrator, reboot). It is donationware and **must never be bundled
   or downloaded by this project**; `--check` detects whether it is
   present and prints these instructions when it is not.
5. Optional: `.\scripts\install-shortcut.ps1` puts "AI Interpreter" in the
   Start Menu and on the Desktop, launching straight into the UI.

First run downloads the models the configured direction needs, pinned to
exact repository revisions:

| Component | Size |
|---|---|
| Silero VAD | 2 MB |
| IndicConformer Tamil STT | ~130 MB |
| Whisper base (English STT + code-switch rescue) | ~140 MB |
| IndicTrans2 Tamil->English | ~820 MB |
| Piper English voice | ~78 MB |
| MMS Tamil voice | ~109 MB |
| IndicTrans2 English->Tamil (only if the reverse direction is used) | ~820 MB |

Roughly **1.3 GB** for the primary direction. Models live in `models/`
inside the project directory (`stt.download_root` moves them).

## 4. Meeting-application setup

In the interpreter (`.\run.ps1 --ui`): Microphone = your headset, Output =
`CABLE Input (VB-Audio Virtual Cable)`, Start. In Teams/Zoom/Meet: select
`CABLE Output (VB-Audio Virtual Cable)` as the microphone. The meeting
hears the translation; verify with Teams' "Make a test call".

## 5. Versioning

Semantic versioning; the single source of truth is `__version__` in
`src/ai_interpreter/__init__.py` (pyproject reads it dynamically). During
construction the minor version tracked the phase; **1.0.0 marks the
completion of the twelve-phase plan**.

## 6. Licensing

The application is MIT. Every dependency and model, with its licence, is
listed in `THIRD-PARTY-NOTICES.md`. What actually constrains deployment:

**The Tamil voice is non-commercial.** `mms-tam` (Facebook MMS,
CC-BY-NC 4.0) is the default Tamil voice because it is the *only* one
runnable under this project's constraints - every alternative is
PyTorch-only (blocked, C6). The restriction is recorded in the model
registry, and a warning is logged on every load. A **commercial**
deployment must remove Tamil speech output (the application falls back to
the designed on-screen captions), or substitute a commercially licensed
voice when one becomes runnable. English output (Piper, MIT) is unaffected.

**VB-CABLE is donationware and may not be redistributed.** Linked, never
bundled, never silently downloaded.

**PySide6/Qt is LGPL-3.0** - used unmodified and dynamically linked, which
a source distribution satisfies by construction.

## 7. Privacy posture

Everything runs locally; the only network access is the explicit,
revision-pinned model download. Meeting audio never leaves the machine.
Transcript text is excluded from log files by default
(`privacy.log_transcripts`). Recordings made by `--record` stay in
`recordings/`, which is git-ignored and excluded from packaging.

## 8. Uninstalling

Delete the project directory (environment, models, logs, recordings all
live inside it), `.\scripts\install-shortcut.ps1 -Remove` for the
shortcuts, and remove `%APPDATA%\HikeHealthGS\ai-interpreter` if personal
configuration overrides were saved. Nothing is written to the registry.

## 9. Release checklist

- [ ] Fast suite and `pytest -m requires_model` both pass
- [ ] ruff, ruff format, mypy clean (`.\scripts\quality.ps1`)
- [ ] Version bumped in `src/ai_interpreter/__init__.py`
- [ ] `requirements.lock.txt` regenerated after any dependency change
- [ ] `THIRD-PARTY-NOTICES.md` still matches `pyproject.toml` and `config/models.yaml`
- [ ] `.\scripts\package.ps1` builds; archive spot-checked for excluded content (no `.env`, no models, no recordings)
- [ ] Fresh-machine install rehearsed: unzip -> bootstrap -Locked -> `--check` -> `--ui`
- [ ] Git tag created
- [ ] (When a code-signing certificate exists) revisit the frozen-EXE decision in §1
