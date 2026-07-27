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

## Audio (Phases 3 and 7)

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
