"""Import-boundary contract (fleet RTM-001).

Rejected requests never import model libraries; valid snapshots still reach them.
"""

import hashlib
import json

import pytest

from geneformer_single_cell_pipeline.pipeline import (
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    GeneformerPipeline,
    GeneVocabulary,
    validate_inputs,
)

_CONFIG = json.dumps({"model_type": "bert", "hidden_size": 768}).encode()
_VOCAB = GeneVocabulary(
    tokens={"<pad>": 0, "<mask>": 1, "<cls>": 2, "<eos>": 3, "ENSG1": 4},
    medians={"ENSG1": 1.0},
    symbol_to_id={},
)


def _snapshot(root, tamper=False):
    (root / "config.json").write_bytes(_CONFIG)
    digest = "0" * 64 if tamper else hashlib.sha256(_CONFIG).hexdigest()
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [{"path": "config.json", "bytes": len(_CONFIG), "sha256": digest}],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_from_pretrained_refuses_without_manifest_before_model_imports(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="no snapshot manifest"):
        GeneformerPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_tampered_snapshot_before_model_imports(tmp_path, forbid_model_imports):
    _snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        GeneformerPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_invalid_cells_are_rejected_before_model_imports(forbid_model_imports):
    with pytest.raises(ValueError, match="at least 10 are required"):
        validate_inputs([{"ENSG1": 5}], _VOCAB)
