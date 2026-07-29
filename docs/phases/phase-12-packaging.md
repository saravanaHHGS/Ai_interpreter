# Phase 12 - Packaging and release

The final phase, and the one whose plan changed most between design and
delivery - for a reason the project itself discovered.

## The decision: a source archive, not a frozen executable

Phase 1 planned PyInstaller + Inno Setup. Phase 12 ships
`scripts\package.ps1` -> `dist\ai-interpreter-<version>.zip` (~0.3 MB)
instead, because of what Smart App Control did on the reference machine
across phases 2-8: it blocked PyTorch's DLLs permanently, blocked mypy's
compiled runtime mid-project, and offers no per-file override. An
unsigned PyInstaller bootloader is precisely the kind of binary it
blocks - the "installer" would fail on exactly the well-managed Windows 11
machines this project targets.

The source distribution sidesteps the problem structurally: every binary
that executes is signed by a publisher SAC already trusts (python.org's
interpreter, PyPI wheels, the Qt Company's PySide6 - verified under
enforced SAC in Phase 8). The application itself is pure Python. The
frozen-EXE route stays on the release checklist, gated on acquiring a
code-signing certificate.

## What shipped

- **`scripts\package.ps1`** - refuses to package unless the fast suite
  passes; stages exactly the reviewed file list (no `.env`, no models, no
  recordings, no venv, no git history, no build artefacts); the archive
  contents were verified entry-by-entry.
- **`scripts\install-shortcut.ps1`** - Start Menu and Desktop shortcuts
  straight into `--ui`, per-user only, nothing in HKLM; `-Remove` deletes
  them. Verified round-trip.
- **`LICENSE`** (MIT) and **`THIRD-PARTY-NOTICES.md`** - every dependency
  and model with its licence; the two constraints that matter are called
  out: the Tamil voice is CC-BY-NC (non-commercial), and VB-CABLE is
  donationware that must never be redistributed.
- **`docs/deployment.md` rewritten** for reality: install steps for a new
  machine, per-model download sizes (~1.3 GB for the primary direction),
  meeting-app setup, privacy posture, uninstall (delete the folder;
  nothing in the registry), and the release checklist.
- **README rewritten** for the finished product.

## Version

**1.0.0** - the twelve-phase plan is complete. Every phase left the
application runnable; this one leaves it installable by someone who was
never part of the construction.
