# Troubleshooting

## Start here

```powershell
.\run.ps1 --check
```

This reports hardware, the selected profile, every configuration source,
package availability, external tools, audio endpoints and privacy settings. It
exits non-zero if anything required is missing. **Include its output in any
bug report.**

If a setting is not behaving as expected:

```powershell
.\run.ps1 --print-config
```

The comment lines at the top list every file that contributed and every
environment override, in precedence order.

---

## Installation and environment

### `running scripts is disabled on this system`

PowerShell blocks scripts by default.

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Applies to your user only and still requires signatures on downloaded scripts.

### `ModuleNotFoundError: No module named 'ai_interpreter'`

Python is running from outside the virtual environment.

- Use `.\run.ps1` rather than calling `python` directly.
- In VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* →
  `.\.venv\Scripts\python.exe`.
- Verify: `.\.venv\Scripts\python.exe -c "import ai_interpreter; print(ai_interpreter.__file__)"`

### `No module named 'pip._internal.cli'`

An interrupted pip self-upgrade corrupted pip inside the virtual environment.
This happens on Windows when antivirus holds a file open during the upgrade.

```powershell
.\scripts\bootstrap.ps1 -Recreate
```

Do not run `pip install --upgrade pip` inside the venv; the bundled pip is
sufficient and the bootstrap script deliberately avoids it.

### `WinError 32: The process cannot access the file`

Something has the file open — usually antivirus, an editor, or a running
Python process. Close editors and terminals holding the project, then retry.
If it persists, add `C:\ai_interpreter` to your antivirus exclusions.

### `py` is not recognised

Python is not installed, or was installed without the launcher. Reinstall from
python.org and tick **Add python.exe to PATH**.

---

## Configuration

### A change to `default.yaml` has no effect

A later layer is overriding it. Precedence, lowest to highest:

1. `config/default.yaml`
2. `config/profiles/<profile>.yaml`
3. `%APPDATA%\HikeHealthGS\ai-interpreter\config.yaml`
4. `AI_INTERPRETER__*` environment variables

Run `.\run.ps1 --print-config` — the header lists exactly which sources
applied. A common cause is editing `default.yaml` when the active profile
overrides that key.

### `Configuration is invalid` with `extra_forbidden`

An unrecognised key, almost always a typo. The message names the exact path,
for example `stt.moel: Extra inputs are not permitted`. Unknown keys are
rejected on purpose: silently ignoring them is how you spend an hour tuning a
setting that was never read.

### `Required configuration file not found`

`config/default.yaml` or a profile file is missing. If you started from a
clone, run `git status` to see whether it was deleted. The error lists the
profiles it did find.

### `privacy.telemetry must be false`

Correct behaviour. The application has no telemetry backend, so the schema
refuses to start rather than accept a setting that implies otherwise. Set it
back to `false`.

### `translation.cache.enabled is true but privacy.cache_translations is false`

The privacy setting is the master switch. Either enable
`privacy.cache_translations` or disable `translation.cache.enabled`. The
combination is rejected because it is ambiguous, not because either value is
wrong.

### Environment variable override is ignored

The format is `AI_INTERPRETER__SECTION__KEY` — **two** underscores between
levels, one inside key names. Correct:

```powershell
$env:AI_INTERPRETER__VAD__MIN_SILENCE_MS = "450"
```

Note also that a variable set with `$env:` lives only in the current
PowerShell window.

---

## Profiles

### The wrong profile was selected

Check the reason:

```powershell
.\run.ps1 --check
```

The *Reason* line explains the choice. Selection rules:

| Condition | Profile |
|---|---|
| NVIDIA GPU with >= 6 GB VRAM | `cuda` |
| >= 6 physical cores, no usable GPU | `cpu_high` |
| Otherwise | `cpu_low` |

Force one:

```powershell
.\run.ps1 --check --profile cpu_high
```

or set `app.profile` in configuration.

### An NVIDIA GPU is present but `cuda` was not chosen

Either the GPU has under 6 GB of video memory (the reason line says so), or
`nvidia-smi` is not on PATH. Check with `nvidia-smi` in a terminal; if it is
missing, reinstall the NVIDIA driver.

---

## Logging

### The log file is empty

Records are flushed by a background listener thread on shutdown. If the
process was killed rather than exiting, the tail can be lost. Use `Ctrl+C`
rather than closing the window.

### Transcript text is missing from the logs

Working as designed. Meeting content is filtered out unless you opt in:

```yaml
privacy:
  log_transcripts: true
```

Turn it off again afterwards, and remember that logs then contain what was
said in your meeting.

### Log files are growing too large

```yaml
logging:
  max_bytes: 5242880   # 5 MB per file
  backup_count: 3      # keep 3 old files
```

Total ceiling is `max_bytes × (backup_count + 1)` per log file.

---

## Native libraries blocked by Windows Smart App Control

### `An Application Control policy has blocked this file`

Windows Smart App Control (or a WDAC policy) is enforced and is blocking an
unsigned native DLL shipped inside a pip package. Check with:

```powershell
(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy").VerifiedAndReputablePolicyState
```

`1` means enforced, `2` evaluation, `0` off.

**Do not turn it off to work around this.** Smart App Control cannot be
re-enabled once disabled without reinstalling Windows, and every library this
project needs works under it.

Known cases, both already handled:

| Library | Effect | Handling |
|---|---|---|
| `hf_xet` | Model download fails | Detected automatically; falls back to plain HTTPS |
| `pysilero-vad` | ggml DLLs blocked | Not used; Silero runs via ONNX Runtime instead |
| `torch` | Blocked outright (WinError 4551) | Constraint C6: only pre-exported ONNX / CTranslate2 models are used |
| `librt` (mypy's runtime) | mypy fails to start | mypy pinned to 1.14.1, installed from source — pure Python, no DLL. `bootstrap.ps1` handles it |

Note that verdicts can **change over time**: `librt` worked for days before
its cloud reputation flipped and it was blocked mid-project. If a previously
working tool suddenly reports `DLL load failed ... An Application Control
policy has blocked this file`, this is what happened.

Verdicts can also be **transient**: `av.filter.loudnorm` (inside
faster-whisper's import chain) was blocked once and importable again twenty
minutes later, with no change on this machine. **Retry once before treating a
new block as permanent.** If Whisper's block ever becomes permanent, switch
the English recogniser to the sherpa-onnx NeMo model with one config line:
`stt.language_models.en: nemo-streaming-en`.

Verified working under enforcement: `onnxruntime`, `sounddevice`, `soundfile`,
`soxr`, `ctranslate2`, `sherpa_onnx`, `onnx`.

If a *new* library is blocked, prefer an ONNX or pure-Python alternative
rather than disabling the policy.

---

## Audio (Phases 3 and 7)

### No audio was captured at all

```powershell
.\run.ps1 --record 10
```

reports `Blocks captured 0`. Causes, in order of likelihood:

1. Windows microphone privacy: Settings → Privacy & security → Microphone →
   allow desktop apps.
2. The device is in use exclusively by another application (Teams, Zoom).
3. The device is disabled in Sound settings.

### The signal is silent or very quiet

`--record` reports a peak below 0.02. Raise the level in Sound settings →
Recording → your microphone → Properties → Levels. If it is already at
maximum, add gain in configuration:

```yaml
audio:
  input:
    gain_db: 10.0
```

Gain is applied after filtering and clips rather than wrapping, so an
excessive value distorts instead of producing noise.

### The signal clips

Peak reaches 1.000. Lower the Windows recording level, or set a negative
`audio.input.gain_db`. Clipped audio transcribes badly.

### Audio was captured but no speech was detected

The level was fine but the detector never triggered. Try, in order:

1. Speak closer to the microphone. Silero is trained on speech, and distant
   room audio scores low by design.
2. Lower the threshold: `vad.threshold: 0.35`.
3. Switch detectors for a comparison: `vad.provider: energy`.

### The application cuts me off mid-sentence

`vad.min_silence_ms` is too short. Raise it to 450–600. The cost is that every
translation arrives correspondingly later — this setting is the largest single
term in the latency budget.

### It waits too long before responding

Lower `vad.min_silence_ms` towards 250. Below about 200 ms, natural pauses
between words start splitting sentences into fragments, which translate badly.

### The first word is cut off

`vad.pre_roll_ms` is too small. Raise it to 400. It costs nothing in latency —
the audio is buffered continuously and used retroactively.

### Blocks are being dropped

`--record` reports a non-zero `Blocks dropped`. The machine could not keep up
and audio was lost. Close other applications. If it persists, reduce
`stt.cpu_threads` so the capture thread is not starved.

### The wrong microphone is being used

```powershell
.\run.ps1 --list-devices
```

The `Would capture from` line shows the resolved device. Pin it explicitly:

```yaml
audio:
  input:
    device: "Internal Microphone"
    host_api: WASAPI
```

The name is matched case-insensitively as a substring.

### A device name looks truncated

MME truncates device names to 31 characters, so
`Internal Microphone (Conexant ISST Audio)` appears as
`Internal Microphone (Conexant I`. This is why `audio.input.host_api` defaults
to `WASAPI`. Do not copy a truncated MME name into configuration.

### `--check` shows no virtual cable

Expected until you install VB-CABLE. See [setup.md](setup.md) section 7. It
requires a reboot.

### Teams cannot see the virtual microphone

1. Confirm the reboot after installing VB-CABLE actually happened.
2. Windows Settings → System → Sound → check *CABLE Output* is not disabled.
3. In Teams: Settings → Devices → Microphone → **CABLE Output**.
4. Fully quit and reopen Teams; it enumerates devices at startup.

### Distorted or robotic audio through the cable

A sample-rate mismatch. Set both endpoints identically:

Sound Control Panel → Playback → *CABLE Input* → Properties → Advanced →
**16 bit, 48000 Hz**. Repeat for Recording → *CABLE Output*.

### The application hears itself and loops

Only possible in bidirectional mode with a single cable. Use the two-cable
topology described in [architecture.md](architecture.md) section 9, or stay in
one-directional mode.

### Bluetooth headset sounds muffled and adds delay

Windows switches Bluetooth headsets into hands-free mode when the microphone
is in use, dropping the audio to 8-16 kHz mono. This hurts both latency and
recognition accuracy. Use a wired or USB headset.

---

## Speech to text (Phase 4)

### The transcript is fluent nonsense in the wrong language

Whisper was forced to decode audio in a language it is not in, and it will
happily invent plausible text rather than fail. The decode language comes from
`app.language_pair.source` unless `stt.language` overrides it.

```powershell
.\run.ps1 --transcribe file.wav --language en
```

Note that `--language` also re-tags the utterances, because an utterance's own
language tag takes precedence over the recogniser's default.

### Transcription is slow

```powershell
.\run.ps1 --benchmark
```

Measured on an Intel i5-7200U (2 physical cores, int8, per utterance):

| Model | 2 threads |
|---|---|
| `tiny` | 0.85 s, noticeably error-prone |
| `base` | 1.66 s, the `cpu_low` default |
| `small` | 5.88 s, unusable |

Decode time is roughly **constant per utterance**, not proportional to its
length, because Whisper pads its encoder to a fixed 30-second window. A
one-word utterance costs about the same as a ten-second one.

To go faster: `stt.model: tiny` (measurable accuracy loss), or a GPU.
Enabling `stt.streaming` does **not** help — see below.

### Setting `stt.streaming: true` made things worse

Expected. Every interim decode costs a full Whisper encoder pass, so streaming
adds CPU load without reducing the delay after you stop speaking. It exists to
show live captions during long sentences. On CPU profiles, leave it off.

### More threads made it slower

Also expected on a 2-core machine. `stt.cpu_threads` should match **physical**
cores, not logical: hyperthread siblings contend for the same execution units,
and CTranslate2's matrix kernels lose more to that than they gain. Measured:
`base` took 1.66 s on 2 threads and 2.07 s on 4.

### Confidence looks low even when the transcript is right

Confidence is `exp(avg_logprob)`, the geometric mean of per-token
probabilities. Correct English decoded as English measured 0.33–0.61 on the
development machine. It is a *relative* signal, not a probability that the
transcript is correct, and its range overlaps with wrong-language output.
Do not set `stt.min_confidence` above about 0.2 without measuring your own
values first, or correct transcripts will be discarded.

### Nothing is transcribed but speech was detected

Whisper returned no segments. Usually the audio is too quiet or too noisy.
Check the level with `--record 10`, then try `--language` explicitly — forcing
the wrong language can also produce an empty result.

### The first transcription is much slower than the rest

The first decode loads the model and initialises kernels. `warmup()` pays this
during startup; commands that report a "Warmup" time have already done so.
1–5 seconds is normal, depending on whether the model file is in the OS cache.

---

## Performance

### Latency is higher than expected

Check the profile first — running `cpu_low` on a capable machine leaves a lot
on the table, and running `cuda` without a GPU is worse.

Then the tuning knobs, in order of impact:

| Setting | Effect |
|---|---|
| `vad.min_silence_ms` | Largest fixed cost. Lower for snappier output, raise if it cuts you off |
| `stt.model` | `small` → `base` is much faster but noticeably less accurate on Tamil |
| `stt.beam_size` | Must stay at 1 for real-time use |
| `pipeline.inference_lane` | `serial` on 2 cores, `parallel` on 6+ |

### High CPU usage / other applications stutter

Reduce `stt.cpu_threads` to leave headroom for audio capture and the UI.
Setting it to the total core count starves the capture thread and causes
dropouts, which sound like clicks.

---

## Reporting a problem

Include:

1. Full output of `.\run.ps1 --check`
2. Output of `.\run.ps1 --print-config`
3. The last ~50 lines of `logs\errors.log`
4. What you did, what you expected, what happened

**Before pasting logs:** if you enabled `privacy.log_transcripts`, the log
contains what was said in your meeting. Review it first.
