"""Unit tests for ONNX metadata patching.

A minimal one-node model is built in memory with the ``onnx`` helpers - no
weights, no torch - so the patcher is exercised against real protobuf files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_interpreter.domain.errors import ModelLoadError
from ai_interpreter.infrastructure.stt.onnx_metadata import ensure_onnx_metadata

pytestmark = pytest.mark.unit

REQUIRED = {"vocab_size": "257", "normalize_type": "per_feature"}


def _write_model(path: Path, metadata: dict[str, str] | None = None) -> Path:
    """Write a minimal valid ONNX model.

    Args:
        path: Destination file.
        metadata: Metadata props to embed, if any.

    Returns:
        The path written.
    """
    from onnx import TensorProto, helper

    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)
    for key, value in (metadata or {}).items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value

    import onnx

    onnx.save(model, str(path))
    return path


def _read_metadata(path: Path) -> dict[str, str]:
    """Read metadata props back from a model file.

    Args:
        path: Model file.

    Returns:
        The embedded metadata.
    """
    import onnx

    model = onnx.load(str(path))
    return {prop.key: prop.value for prop in model.metadata_props}


class TestEnsureOnnxMetadata:
    """Stamping required keys into a copy."""

    def test_patches_a_model_missing_everything(self, tmp_path: Path) -> None:
        source = _write_model(tmp_path / "model.onnx")
        patched = ensure_onnx_metadata(source, tmp_path / "patched", REQUIRED)

        assert patched != source
        assert _read_metadata(patched) == REQUIRED

    def test_source_is_never_mutated(self, tmp_path: Path) -> None:
        # The Hugging Face cache must keep holding exactly what was
        # downloaded.
        source = _write_model(tmp_path / "model.onnx")
        ensure_onnx_metadata(source, tmp_path / "patched", REQUIRED)

        assert _read_metadata(source) == {}

    def test_complete_model_is_used_directly(self, tmp_path: Path) -> None:
        # The csukuangfj exports already carry their metadata; copying
        # 130 MB for nothing would be pure waste.
        source = _write_model(tmp_path / "model.onnx", metadata=REQUIRED)
        result = ensure_onnx_metadata(source, tmp_path / "patched", REQUIRED)

        assert result == source
        assert not (tmp_path / "patched").exists()

    def test_existing_keys_are_preserved_not_overwritten(self, tmp_path: Path) -> None:
        source = _write_model(
            tmp_path / "model.onnx", metadata={"vocab_size": "999", "extra": "kept"}
        )
        patched = ensure_onnx_metadata(source, tmp_path / "patched", REQUIRED)

        metadata = _read_metadata(patched)
        assert metadata["vocab_size"] == "999"
        assert metadata["extra"] == "kept"
        assert metadata["normalize_type"] == "per_feature"

    def test_second_call_reuses_the_patched_copy(self, tmp_path: Path) -> None:
        source = _write_model(tmp_path / "model.onnx")
        first = ensure_onnx_metadata(source, tmp_path / "patched", REQUIRED)
        stamp = first.stat().st_mtime_ns
        second = ensure_onnx_metadata(source, tmp_path / "patched", REQUIRED)

        assert second == first
        assert second.stat().st_mtime_ns == stamp

    def test_no_requirements_is_a_no_op(self, tmp_path: Path) -> None:
        source = _write_model(tmp_path / "model.onnx")
        assert ensure_onnx_metadata(source, tmp_path / "patched", {}) == source

    def test_unreadable_model_reports_clearly(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.onnx"
        broken.write_bytes(b"not a protobuf")

        with pytest.raises(ModelLoadError, match="Could not read"):
            ensure_onnx_metadata(broken, tmp_path / "patched", REQUIRED)
