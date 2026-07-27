# Phase 2 — Project Foundation

**Status:** complete
**Version:** 0.2.0

Builds the skeleton every later phase attaches to: project layout, virtual
environment, typed configuration, logging, dependency injection, hardware
detection and the test harness. No machine learning yet — and the application
runs.

---

## Verification

```powershell
.\run.ps1 --check
```

Expected on a two-core laptop with no NVIDIA GPU:

```
Selected profile           cpu_low
Reason                     no NVIDIA GPU and 2 physical cores, so small
                           quantised models run one at a time
Speech-to-text             faster_whisper / small
Inference lane             serial
...
All required checks passed.
```

Two warnings are expected and correct at this stage: FFmpeg (needed from
Phase 3) and no virtual audio cable (needed from Phase 7).

```powershell
.\scripts\quality.ps1
```

Expected: formatting clean, lint clean, mypy clean, **134 tests passed**.

---

## Files

### Root

| File | Purpose |
|---|---|
| `pyproject.toml` | Metadata, dependencies, ruff/mypy/pytest configuration. Single source of truth for dependency ranges |
| `requirements.txt` | `-e .` — installs the project, so it cannot drift from pyproject |
| `requirements-dev.txt` | `-e .[dev]` — adds test and lint tools |
| `requirements.lock.txt` | Exact pins from `pip freeze`, for reproducible installs |
| `.env.example` | Secrets template. Copied to `.env` by bootstrap |
| `.gitignore` | Excludes `.venv`, `logs/`, `models/`, `recordings/`, `.env` |
| `.python-version` | Records 3.12 |
| `run.ps1` | Launcher. Calls the venv interpreter without activating it |

### Configuration

| File | Purpose |
|---|---|
| `config/default.yaml` | Every default value, commented throughout |
| `config/profiles/cpu_low.yaml` | 2-4 cores, no GPU — this machine |
| `config/profiles/cpu_high.yaml` | 6+ cores, no GPU |
| `config/profiles/cuda.yaml` | NVIDIA GPU, 6+ GB VRAM |

### Domain — `src/ai_interpreter/domain/`

| File | Contents |
|---|---|
| `value_objects.py` | `LanguageCode`, `LanguagePair`, `SampleRate`, `Confidence`, `StageTiming`, enums |
| `entities.py` | `AudioFrame`, `Utterance`, `Transcript`, `Translation`, `SpeechAudio`, `VoiceInfo`, `DeviceInfo`, `HardwareInfo`, `GpuInfo`, `ModelDescriptor` |
| `ports.py` | All 13 `Protocol` interfaces |
| `errors.py` | Error hierarchy rooted at `InterpreterError` |

### Infrastructure — `src/ai_interpreter/infrastructure/`

| File | Contents |
|---|---|
| `paths.py` | Filesystem layout; `AI_INTERPRETER_HOME` and `AI_INTERPRETER_USER_HOME` overrides |
| `config/settings.py` | Pydantic schema; `extra="forbid"`, `frozen=True` |
| `config/loader.py` | Four-layer merge, env parsing, validation with readable errors |
| `config/secrets.py` | `.env` loading with `SecretStr` redaction |
| `logging/setup.py` | Queue-backed rotating logs, transcript filter, UTF-8 handling |
| `system/hardware.py` | CPU, RAM, disk and NVIDIA GPU detection |
| `system/audio_endpoints.py` | Windows registry audio endpoint enumeration |

### Application and composition root

| File | Contents |
|---|---|
| `application/services/profile_selector.py` | Hardware → profile mapping, pure logic |
| `app/container.py` | The composition root — the only file that wires concretes to ports |
| `cli.py` | `--check`, `--print-config`, `--version`, `--profile` |

---

## Dependencies and why

| Package | Why this one | Alternative rejected |
|---|---|---|
| `numpy` | Audio buffers | none viable |
| `pydantic` + `pydantic-settings` | Typed, validated config; fails fast with readable errors | plain dicts — no validation |
| `PyYAML` | Config format users hand-edit; comments matter | TOML — weaker nesting |
| `platformdirs` | Correct `%APPDATA%` paths | hard-coded paths |
| `psutil` | Cross-platform CPU/RAM detection | `os.cpu_count()` cannot see physical cores |
| `pytest` | Fixtures, parametrisation, markers | unittest — more ceremony |
| `ruff` | Replaces flake8 + isort + black + pyupgrade, ~100x faster | the four separate tools |
| `mypy` | Static type checking | pyright — mypy has the pydantic plugin |

Logging deliberately uses the **standard library** rather than `loguru`:
`transformers`, `torch` and `faster-whisper` all log through `logging`, so
configuring it directly captures their output too, with one less dependency.

---

## Decisions worth understanding

### Unknown configuration keys are errors

`extra="forbid"` means `min_silense_ms` fails at startup naming the exact key,
instead of being ignored while you wonder why your tuning did nothing.

### Defaults live in YAML, not Python

Most schema fields have no default. If `default.yaml` omits a key, startup
fails. One source of truth, and the user can read it.

### Telemetry is rejected, not merely disabled

The schema raises if `privacy.telemetry` is true. A setting that cannot be
switched on cannot be switched on by accident.

### Logging sits behind a queue

`RotatingFileHandler` writes synchronously and renames files during rollover.
On the audio thread that means dropped samples and an audible click.
`QueueHandler` makes logging a near-instant in-memory append; a listener
thread does the disk work.

### The console is forced to UTF-8

The Windows console defaults to a legacy code page. Printing Tamil or Hindi
raises `UnicodeEncodeError` from inside the logging machinery, which reads
like a crash in the model. One of the most common failures in Indic-language
Python applications on Windows.

### Dependency injection is hand-written

`container.py` is ~120 lines and fully type-checked. A DI framework would
resolve the graph by reflection, defeating mypy and turning wiring mistakes
into runtime exceptions.

---

## A real bug found while building this phase

The first full test run **wrote into the developer's real `%APPDATA%`
directory**. `ApplicationPaths` accepted a temporary project root but still
resolved user directories to the real user profile, so a test that saved a
user override modified actual configuration — and made a later test fail
depending on run order.

The fix was structural, not a patch to that one test:

1. `ApplicationPaths` honours `AI_INTERPRETER_USER_HOME`.
2. An **autouse** fixture sets it for every test, so no test can reach the
   real profile even if someone forgets.
3. The stray file that had been created was deleted.

The environment variable is not test-only scaffolding — the Phase 12 portable
build needs exactly the same capability.

---

## Deviations from the Phase 1 design

| Phase 1 said | Reality | Why |
|---|---|---|
| Domain has zero dependencies | Domain imports numpy | Audio buffers cannot be `list[float]` |
| `presentation/` package created | Deferred to Phase 8 | An empty package would be a placeholder |
| `config/models.yaml` in Phase 2 | Deferred to Phase 4 | It needs real repository revisions and hashes, which are only known once models are chosen and downloaded |

---

## What Phase 3 adds

Audio capture: `sounddevice`, device enumeration and selection, ring buffer,
resampling to 16 kHz, Silero VAD, utterance segmentation, background capture
thread, and test recordings saved to `recordings/`.

New dependencies: `sounddevice`, `soundfile`, `soxr`, `onnxruntime`.

The `AudioSource`, `DeviceEnumerator` and `VoiceActivityDetector` ports it
implements already exist in `domain/ports.py`.
