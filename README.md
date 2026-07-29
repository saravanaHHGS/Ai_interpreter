# AI Interpreter

Real-time speech-to-speech translation for Windows meeting applications.

You speak Tamil. Microsoft Teams hears English. It works with Teams, Zoom
and Google Meet because it outputs through a **virtual microphone** rather
than integrating with any particular meeting platform - and it handles the
way people actually talk, mixing English technical terms and product names
into Tamil sentences.

Everything runs locally on an ordinary 2-core CPU. No cloud service, no API
key, no GPU, no audio leaves your machine.

---

## Status: feature-complete (v1.0)

All twelve construction phases are delivered. Highlights, each measured on
the reference machine (an Intel i5-7200U, 2 cores):

- **Live interpretation** at a mean of ~1.7 s from end-of-sentence to the
  first translated audio, with per-utterance latency reporting.
- **Code-switch understanding**: "VALD assessment முடிஞ்சுச்சு" is repaired
  by fusing two recognisers' views of the same audio word-by-word, with
  hotword biasing, a phonotactic detector, and a user-editable glossary as
  layered defences. See [docs/code-switching.md](docs/code-switching.md).
- **A desktop interface** (`--ui`): device pickers, Start/Stop, live
  captions with a streaming partial line, latency readout. Second Start is
  instant (0.01 s - models stay warm).
- **A streaming lane** for long speech: 20.7 s of continuous talk produced
  its final transcript 0.64 s after the speaker stopped, against 13.4 s for
  the offline decode.
- **640 tests** (630 fast + 10 real-model regressions over ground-truthed
  recorded speech), ruff and strict mypy clean.

---

## Quick start

```powershell
# 1. Install Python 3.12+ from python.org (tick "Add python.exe to PATH")
# 2. Unzip the release (or git clone) to C:\ai_interpreter, then:
cd C:\ai_interpreter
.\scripts\bootstrap.ps1 -Locked     # environment + dependency install + doctor
# 3. Install VB-CABLE from https://vb-audio.com/Cable/ (see docs/setup.md)

.\run.ps1 --ui                      # the desktop interface
```

In the interpreter: Microphone = your headset, Output = `CABLE Input`,
press **Start**. In Teams: Settings -> Devices -> Microphone =
`CABLE Output`. The meeting now hears your sentences in English.
Optional: `.\scripts\install-shortcut.ps1` adds Start Menu and Desktop
shortcuts.

First run downloads ~1.3 GB of models (exact revisions, verified); see
[docs/deployment.md](docs/deployment.md).

---

## How it works

```
mic -> VAD -> segmentation -> Tamil STT (IndicConformer) ---+
                |                                           |  word-level fusion
                +-> streaming partial captions              |  for mixed sentences
                          English STT (Whisper + hotwords) -+
                                        |
                       glossary -> IndicTrans2 translation -> Piper voice
                                        |
                              CABLE Input (virtual microphone) -> Teams
```

Every stage is behind a typed port (Clean Architecture, four layers, a
hand-written composition root), every model was chosen by benchmarking on
the target CPU, and every design decision that came from a measurement is
written down next to the code it shaped.

---

## Requirements

| | Minimum (verified) | Notes |
|---|---|---|
| OS | Windows 11 64-bit | Smart App Control supported - see docs/deployment.md |
| Python | 3.12 | |
| CPU | 2 cores | the reference machine; more cores help |
| RAM | 8 GB | ~2.5 GB in use while interpreting |
| GPU | none | architecture keeps a CUDA profile for the future |
| Disk | 5 GB free | environment + models |

---

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Layers, ports, constraints, latency budget |
| [docs/setup.md](docs/setup.md) | Installing everything from zero |
| [docs/code-switching.md](docs/code-switching.md) | The mixed-language design and its measurements |
| [docs/testing.md](docs/testing.md) | Running and writing tests |
| [docs/deployment.md](docs/deployment.md) | Packaging, install, licensing |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Symptoms, causes, fixes |
| [docs/phases/](docs/phases/) | What each phase delivered and how it was verified |

---

## Command reference

```powershell
.\run.ps1 --ui                       # desktop interface (recommended)
.\run.ps1 --interpret 60             # live interpretation in the console
.\run.ps1 --interpret 300 --out "CABLE Input"   # live into Teams' microphone
.\run.ps1 --wav recording.wav        # reproducible pipeline run over a file
.\run.ps1 --check                    # environment doctor
.\run.ps1 --list-devices             # every audio input and output device
.\run.ps1 --record 10                # capture, detect speech, save WAVs
.\run.ps1 --listen 20                # live speech to text only
.\run.ps1 --transcribe file.wav      # transcribe a file with timings
.\run.ps1 --translate "நாளைக்கு என்ன திட்டம்?"          # Tamil -> English
.\run.ps1 --speak "Hello Teams" --language en --out "CABLE Input"
.\run.ps1 --benchmark                # measure decode time on your machine
.\run.ps1 --print-config             # effective configuration as YAML

.\scripts\bootstrap.ps1 -Locked      # set up or repair the environment
.\scripts\quality.ps1                # lint, types and tests
.\scripts\package.ps1                # build the distributable archive
.\scripts\install-shortcut.ps1       # Start Menu / Desktop shortcuts
```

Tests: `pytest` (fast suite, ~10 s) and `pytest -m requires_model` (real
neural networks against ground-truthed speech, ~1 min).

---

## Licence

MIT for this project's code ([LICENSE](LICENSE)). Dependency and model
licences are listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
One default is non-commercial: the Tamil *voice* (Facebook MMS,
CC-BY-NC 4.0) - the only Tamil voice runnable on CPU-only Windows today.
Commercial deployments must fall back to Tamil captions or substitute a
licensed voice; everything else in the default configuration is
commercially usable. VB-CABLE is donationware, installed by the user, and
never redistributed.
