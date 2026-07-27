# Deployment

Packaging is built in Phase 12. This document records the decisions that
Phase 12 depends on — particularly licensing, which constrains what may be
shipped and is far cheaper to get right now than to discover at the end.

---

## 1. Planned artefacts

| Artefact | Tool | Purpose |
|---|---|---|
| Windows installer | Inno Setup | Start Menu entry, uninstaller, per-user install |
| Portable build | PyInstaller | Runs from a folder or USB stick, no installation |
| Update check | Custom, opt-in | Compares versions against a release feed |

Both builds are produced from the same PyInstaller output. The portable
version sets `AI_INTERPRETER_HOME` and `AI_INTERPRETER_USER_HOME` so that
configuration, logs and models live beside the executable and nothing is
written to the user profile.

Those two environment variables already exist and are already tested — the
portable build is a configuration of the application, not a special mode in
the code.

---

## 2. Versioning

Semantic versioning. The single source of truth is `__version__` in
`src/ai_interpreter/__init__.py`; `pyproject.toml` reads it via
`[tool.setuptools.dynamic]`, so the two cannot disagree.

During construction the minor version tracks the phase: `0.2.0` is Phase 2.
The first feature-complete release is `1.0.0`.

---

## 3. Licensing

### This project

MIT.

### Dependencies

| Component | Licence | Commercial use | Notes |
|---|---|---|---|
| PySide6 | LGPLv3 | Yes | Must be dynamically linked — PyInstaller's default is fine |
| numpy, pydantic, PyYAML | BSD / MIT | Yes | |
| sounddevice / PortAudio | MIT | Yes | |
| onnxruntime | MIT | Yes | |
| faster-whisper, CTranslate2 | MIT | Yes | |
| Whisper weights | MIT | Yes | |
| Silero VAD | MIT | Yes | |
| IndicTrans2 | MIT | Yes | Default translator |
| Piper | MIT | Yes | Default synthesizer |
| Kokoro-82M | Apache-2.0 | Yes | CUDA profile synthesizer |
| Parakeet TDT | CC-BY-4.0 | Yes | Requires attribution |
| FFmpeg | LGPL / GPL build-dependent | Care needed | Not bundled; installed separately |

### Deliberately excluded from defaults

| Component | Licence | Why it is not a default |
|---|---|---|
| **NLLB-200** | CC-BY-NC | **Non-commercial only** |
| **Coqui XTTS-v2** | CPML | **Non-commercial only** |

Both remain available as opt-in adapters. Selecting one shows a licence
warning in the UI. They are never selected automatically and never appear in a
profile file.

### VB-CABLE — the one that catches people out

VB-CABLE is **donationware and may not be redistributed.** The installer must
not bundle it. Phase 12 will:

1. Detect whether a virtual cable is present (the code for this already
   exists and runs in `--check`).
2. If absent, link to <https://vb-audio.com/Cable/> with instructions.
3. Never download or install it silently.

### Attribution

The installer and the About page will carry a `THIRD-PARTY-NOTICES.md`
listing every dependency and model with its licence, including the CC-BY-4.0
attribution Parakeet requires.

---

## 4. Model distribution

Models are **not** bundled. The installer is a few tens of megabytes; models
total 1.5-12 GB depending on profile.

On first run the application downloads only what the selected profile needs:

| Profile | Approximate download |
|---|---|
| `cpu_low` | ~1.5 GB |
| `cpu_high` | ~3 GB |
| `cuda` | ~6-12 GB |

Every model is pinned to an exact repository revision and verified after
download. Tracking a moving branch would let upstream silently change the
weights a user is running — a supply-chain risk with no upside.

---

## 5. Build process (Phase 12)

```powershell
.\scripts\bootstrap.ps1 -Locked   # exact pinned versions, reproducible
.\scripts\quality.ps1             # all gates must pass
.\scripts\build.ps1               # PyInstaller + Inno Setup   (Phase 12)
```

`-Locked` uses `requirements.lock.txt` rather than resolving ranges, so the
shipped binary contains exactly the versions that were tested.

---

## 6. Release checklist (Phase 12)

- [ ] All quality gates pass
- [ ] Version bumped in `src/ai_interpreter/__init__.py`
- [ ] `requirements.lock.txt` regenerated
- [ ] `THIRD-PARTY-NOTICES.md` regenerated
- [ ] Release notes written
- [ ] Installer tested on a clean Windows 11 machine with no Python installed
- [ ] Portable build tested from a folder with no write access to the profile
- [ ] Uninstaller leaves no orphaned files or registry keys
- [ ] Git tag created

---

## 7. Security considerations for distribution

| Risk | Mitigation |
|---|---|
| Unsigned binary triggers SmartScreen | Code-signing certificate, or documented "More info → Run anyway" |
| Tampered model weights | Exact revision pinning plus hash verification |
| Credentials in a shipped build | `.env` is git-ignored and excluded from the PyInstaller spec |
| Log files containing meeting content | Transcripts are filtered out of logs by default |
| Unexpected network access | No runtime network calls except explicit model downloads |
