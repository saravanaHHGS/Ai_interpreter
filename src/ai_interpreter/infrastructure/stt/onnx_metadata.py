"""Stamping required metadata into ONNX model files.

sherpa-onnx identifies a NeMo CTC model by metadata embedded in the ONNX file
itself - ``vocab_size``, ``normalize_type``, ``subsampling_factor`` and so on.
The AI4Bharat IndicConformer export published by OpenVoiceOS ships without
that metadata, so sherpa-onnx refuses to load it:

    'vocab_size' does not exist in the metadata

The weights are fine; only the label is missing. This module stamps the
required keys into a *copy* of the model. The Hugging Face cache is never
mutated: its files are content-addressed, and editing them in place would make
the cache lie about what it holds.

Patching runs once per model. The patched copy is the done-marker: if it
exists, nothing is loaded or written. Models that already carry the required
keys (the csukuangfj exports, which are made for sherpa-onnx) are used
directly from the cache with no copy at all.

This exists because of constraint C6: PyTorch is blocked on the target
machine, so re-exporting the model properly is impossible. Editing metadata
with the pure-protobuf ``onnx`` package is the only local option, and happily
it is also the smallest one.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ai_interpreter.domain.errors import ModelLoadError

__all__ = ["ensure_onnx_metadata"]

logger = logging.getLogger(__name__)


def ensure_onnx_metadata(
    source: Path,
    patched_dir: Path,
    required: dict[str, str],
) -> Path:
    """Return a model file that carries the required metadata.

    Args:
        source: The downloaded model file.
        patched_dir: Directory for the patched copy, created if needed.
        required: Metadata keys and values the runtime needs. Existing keys in
            the source are kept; only missing ones are added.

    Returns:
        ``source`` itself when it already has every required key, otherwise
        the path of a patched copy inside ``patched_dir``.

    Raises:
        ModelLoadError: If the model cannot be read or the copy written.
    """
    if not required:
        return source

    patched = patched_dir / source.name
    if patched.is_file():
        return patched

    try:
        import onnx

        model = onnx.load(str(source))
    except Exception as exc:
        msg = f"Could not read the ONNX model {source}: {exc}"
        raise ModelLoadError(msg) from exc

    present = {prop.key for prop in model.metadata_props}
    missing = {key: value for key, value in required.items() if key not in present}
    if not missing:
        logger.debug("%s already carries the required metadata", source.name)
        return source

    for key, value in missing.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value

    try:
        patched_dir.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(patched))
    except Exception as exc:
        msg = f"Could not write the patched model {patched}: {exc}"
        raise ModelLoadError(msg) from exc

    logger.info(
        "Stamped %d metadata key(s) into a copy of %s: %s",
        len(missing),
        source.name,
        ", ".join(sorted(missing)),
    )
    return patched
