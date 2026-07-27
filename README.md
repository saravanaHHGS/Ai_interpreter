# AI Interpreter

Real-time speech-to-speech translation for Windows meeting applications.

You speak Tamil. Microsoft Teams hears English. It works with Teams, Zoom and
Google Meet because it outputs through a **virtual microphone** rather than
integrating with any particular meeting platform.

Everything runs locally. No cloud service, no API key, no audio leaves your
machine.

---

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 1 | Architecture design | Complete |
| 2 | Project foundation: config, logging, DI, tests | Complete |
| 3 | Audio capture, VAD, device selection | **Complete** |
| 4 | Speech to text | Not started |
| 5 | Translation engine | Not started |
| 6 | Text to speech | Not started |
| 7 | Virtual microphone routing | Not started |
| 8 | PySide6 desktop interface | Not started |
| 9 | Streaming pipeline | Not started |
| 10 | Performance optimisation | Not started |
| 11 | Test suite expansion | Not started |
| 12 | Packaging and installer | Not started |

The application is runnable after every phase.

---

## Quick start

```powershell
git clone <repository-url> C:\ai_interpreter
cd C:\ai_interpreter
.\scripts\bootstrap.ps1
```

That creates the virtual environment, installs everything, and runs the
environment check. Then:

```powershell
.\run.ps1 --check           # verify the environment
.\run.ps1 --print-config    # show the settings actually in use
```

Full instructions, including what to install and why: [docs/setup.md](docs/setup.md).

---

## What is in the box today

- **Clean Architecture** in four layers with dependency injection throughout.
- **Typed configuration** from layered YAML with schema validation; an
  unrecognised key stops startup instead of being silently ignored.
- **Hardware profiles** selected automatically. The same code runs small
  quantised models on a two-core laptop and full-quality models on an RTX GPU.
- **Non-blocking rotating logs** that never stall the audio thread.
- **Privacy by construction**: conversation text is kept out of log files
  unless you explicitly opt in, nothing is persisted by default, and no
  telemetry exists.
- **Microphone capture** with device selection, WASAPI preference, stateful
  resampling to 16 kHz, high-pass filtering, and a drop-oldest callback buffer.
- **Neural voice activity detection** (Silero via ONNX Runtime, 0.5 ms per
  32 ms frame) with an adaptive energy detector as a model-free fallback.
- **Utterance segmentation** with pre-roll, so the first syllable is never
  clipped, plus test recordings you can listen to.
- **289 tests**, plus lint and strict type checking, all passing.

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 11 64-bit | Windows 11 64-bit |
| Python | 3.12 | 3.12 |
| CPU | 2 cores | 6+ cores |
| RAM | 8 GB | 16 GB |
| GPU | none (CPU fallback) | NVIDIA RTX, 6+ GB VRAM |
| Disk | 12 GB free | 20 GB free |

Latency depends heavily on hardware. On an RTX GPU the design target is
roughly 0.7-0.9 s from the end of your sentence to the first translated audio.
On a two-core laptop, expect 1.5-2.5 s. Real measurements are produced by the
benchmarks in Phases 4, 6 and 10 — the numbers above are design targets, not
results.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Layers, ports, diagrams, latency budget, design decisions |
| [docs/setup.md](docs/setup.md) | Installing everything from zero, with each command explained |
| [docs/testing.md](docs/testing.md) | How to run and write tests |
| [docs/deployment.md](docs/deployment.md) | Packaging, licensing, distribution |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Symptoms, causes, fixes |
| [docs/phases/](docs/phases/) | What each phase delivered and how to verify it |

---

## Command reference

```powershell
.\run.ps1 --check                    # environment doctor; non-zero exit on failure
.\run.ps1 --list-devices             # every audio input and output device
.\run.ps1 --record 10                # capture 10 s, detect speech, save WAV files
.\run.ps1 --record 10 --device "CABLE Output"   # capture from a named device
.\run.ps1 --print-config             # effective configuration as YAML
.\run.ps1 --print-config --profile cuda   # preview another hardware profile
.\run.ps1 --version
.\run.ps1 --help

.\scripts\bootstrap.ps1              # set up or repair the environment
.\scripts\bootstrap.ps1 -Locked      # install exact pinned versions
.\scripts\quality.ps1                # lint, types and tests
.\scripts\quality.ps1 -Fix           # auto-fix formatting and safe lint issues
```

---

## Licence

MIT for this project's own code. Model and dependency licences differ and are
listed in [docs/deployment.md](docs/deployment.md); two of them are
non-commercial and are deliberately never used as defaults.
