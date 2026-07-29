# Third-party notices

AI Interpreter is MIT-licensed (see `LICENSE`). It stands on the following
third-party software and models, each under its own licence. Nothing below
is bundled in the distribution archive; dependencies install from PyPI and
models download from Hugging Face on first use, pinned to exact revisions.

## Python dependencies

| Component | Licence | Notes |
|---|---|---|
| numpy | BSD-3-Clause | |
| pydantic, pydantic-settings | MIT | |
| PyYAML | MIT | |
| platformdirs | MIT | |
| psutil | BSD-3-Clause | |
| sounddevice / PortAudio | MIT | |
| soundfile / libsndfile | BSD-3-Clause / LGPL-2.1 | dynamically linked |
| soxr | LGPL-2.1 | dynamically linked |
| onnxruntime | MIT | |
| huggingface-hub | Apache-2.0 | |
| faster-whisper | MIT | |
| CTranslate2 | MIT | |
| sherpa-onnx | Apache-2.0 | |
| onnx | Apache-2.0 | |
| sentencepiece | Apache-2.0 | |
| PySide6 / Qt | LGPL-3.0 | dynamically linked, unmodified |

## Models (downloaded on first use, never redistributed by this project)

| Model | Source | Licence | Commercial use |
|---|---|---|---|
| Silero VAD | onnx-community/silero-vad | MIT | Yes |
| Whisper (tiny/base/small) | Systran/faster-whisper-* | MIT | Yes |
| IndicConformer Tamil | OpenVoiceOS/ai4bharat-indicconformer-ta-onnx | MIT (AI4Bharat release) | Yes |
| IndicTrans2 200M (both directions) | adalat-ai/ct2-rotary-indictrans2-* | MIT (AI4Bharat release) | Yes |
| Piper en_US voices | csukuangfj/vits-piper-en_US-* | MIT | Yes |
| **MMS Tamil voice (`mms-tam`)** | willwade/mms-tts-multilingual-models-onnx | **CC-BY-NC 4.0** | **NO - non-commercial only** |

The MMS Tamil voice is the only non-commercial component in the default
configuration, kept because it is the only Tamil voice runnable under this
project's hardware constraints. Commercial deployments must disable Tamil
speech output (the application falls back to on-screen captions), or
substitute a commercially licensed voice. See `docs/deployment.md`.

## Not bundled, not redistributable

**VB-CABLE** (VB-Audio Software) is donationware and may not be
redistributed. This project never bundles or downloads it; users install it
themselves from <https://vb-audio.com/Cable/>.
