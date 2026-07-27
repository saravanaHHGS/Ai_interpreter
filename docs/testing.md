# Testing

## Running the tests

```powershell
.\scripts\quality.ps1              # everything: format, lint, types, tests
.\scripts\quality.ps1 -Coverage    # with a coverage report
.\scripts\quality.ps1 -Fix         # auto-fix formatting and safe lint issues
```

Or individually:

```powershell
.\.venv\Scripts\python.exe -m pytest -q                    # all tests
.\.venv\Scripts\python.exe -m pytest tests/unit -q         # unit tests only
.\.venv\Scripts\python.exe -m pytest -m unit               # by marker
.\.venv\Scripts\python.exe -m pytest -k config -v          # by name
.\.venv\Scripts\python.exe -m pytest --lf                  # only last failures
.\.venv\Scripts\python.exe -m mypy                         # static types
.\.venv\Scripts\python.exe -m ruff check .                 # lint
```

---

## Current state

| Gate | Status |
|---|---|
| `ruff format --check` | 34 files formatted |
| `ruff check` | passing |
| `mypy` (strict-leaning) | no issues in 24 source files |
| `pytest` | **134 passed** |

Runtime: under 5 seconds for the whole suite. That is deliberate. A suite you
run after every edit must be fast enough that you actually do.

---

## Layout

```
tests/
├── conftest.py          shared fixtures
├── unit/                fast, isolated, no I/O beyond a temp directory
├── integration/         several real components wired together
├── performance/         latency benchmarks            (Phase 10)
├── stress/              sustained load                (Phase 11)
└── fixtures/audio/      short ta/hi/en clips          (Phase 3)
```

### Markers

Declared in `pyproject.toml` and enforced with `--strict-markers`, so a typo
in a marker name fails instead of silently doing nothing.

| Marker | Meaning |
|---|---|
| `unit` | Fast, isolated |
| `integration` | Multiple real components |
| `performance` | Latency or throughput measurement |
| `stress` | Long-running load |
| `requires_audio` | Needs a real audio device |
| `requires_model` | Needs downloaded weights |

The last two exist so continuous integration can skip them:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not requires_audio and not requires_model"
```

---

## What is covered today

| Area | Tests | What is verified |
|---|---|---|
| Value objects | 22 | Validation, normalisation, conversions, hashing |
| Entities | 18 | Audio buffer contracts, time ranges, GPU reporting |
| Config loader | 26 | Layer precedence, env parsing, every failure mode |
| Profile selection | 7 | Each hardware class maps to the right tier |
| Logging | 11 | Rotation, levels, privacy filtering, Unicode, idempotency |
| Paths | 12 | Root precedence, directory creation, user-profile isolation |
| Secrets | 7 | Loading, redaction in `repr` |
| Container + CLI | 15 | Full startup, profile override, all commands |

---

## Fixtures

Defined in `tests/conftest.py`.

| Fixture | Provides |
|---|---|
| `isolated_user_directories` | **autouse** — redirects `%APPDATA%` to a temp dir |
| `project_root` | Temp project root with real config copied in |
| `paths` | `ApplicationPaths` rooted there, directories created |
| `cpu_only_hardware` | 2-core laptop, no GPU |
| `multicore_cpu_hardware` | 8-core desktop, no GPU |
| `cuda_hardware` | RTX 3060, 12 GB |
| `small_gpu_hardware` | GTX 1650, 4 GB — too small, must fall back |

### Why `isolated_user_directories` is autouse

While building Phase 2, a test that saved a user override wrote into the
developer's **real** `%APPDATA%` directory. It changed the developer's own
configuration and made a later test fail depending on run order.

The fix was structural rather than a patch to that one test: `ApplicationPaths`
honours an `AI_INTERPRETER_USER_HOME` environment variable, and an autouse
fixture sets it for every test. No test can reach the real user profile even
if someone forgets.

That environment variable is not test-only scaffolding — the Phase 12 portable
build uses it to keep everything beside the executable.

---

## Conventions

**Naming.** `test_<what>_<expected outcome>`, e.g.
`test_rejects_identical_language_pair`. The name should read as a sentence and
tell you what broke without opening the file.

**Grouping.** Related tests share a class with a one-line docstring. Classes
are grouping only — no `setUp`, no inheritance.

**Fakes over mocks.** Ports are `Protocol`s, so a test double is an ordinary
class with the right methods:

```python
class FakeTranslator:
    """Returns a fixed translation without loading a model."""

    model_id = "fake"

    def supports(self, pair: LanguagePair) -> bool:
        return True

    def translate(self, text: str, pair: LanguagePair) -> Translation:
        return Translation(
            utterance_id=UtteranceId("test"),
            source_text=text,
            translated_text="translated",
            pair=pair,
        )

    def warmup(self) -> None: ...
    def close(self) -> None: ...
```

No mock library, no base class, no registration. If the methods are wrong,
mypy says so.

**Test the failure paths.** Roughly half the configuration tests assert that
bad input is *rejected with a useful message*. A validator nobody tested is a
validator that does not work.

**No sleeping.** A test that calls `time.sleep` to wait for a thread is
flaky. Use events and explicit synchronisation.

---

## Adding tests for a new phase

1. Put unit tests in `tests/unit/test_<module>.py`.
2. Put wiring tests in `tests/integration/`.
3. Mark anything needing hardware with `@pytest.mark.requires_audio` or
   `@pytest.mark.requires_model`.
4. For a new port, write the fake first — if the fake is awkward, the port is
   badly designed, and it is much cheaper to find that out now.
5. Run `.\scripts\quality.ps1` before committing.

---

## Coming in later phases

| Phase | Additions |
|---|---|
| 3 | WAV-file `AudioSource`, so the pipeline is testable with no microphone |
| 4 | Accuracy tests against reference transcripts; latency benchmarks |
| 5 | Translation quality checks on a fixed ta/hi/en phrase set |
| 6 | Synthesis latency; the Tamil voice decision gate |
| 9 | Cancellation, backpressure and retry behaviour under load |
| 10 | EOU→FTS benchmark producing the real latency numbers |
| 11 | Stress tests: hours of continuous audio, device disconnection mid-session |
