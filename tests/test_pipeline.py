"""Offline pipeline tests: identity, snapshot verification/staging, restricted unpickling,
rank-value encoding, input checks, and the classify/artifact contracts with injected backends."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path

import pytest

from geneformer_single_cell_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    GENE_MEDIAN_DICTIONARY_FILE,
    HIDDEN_SIZE,
    MANIFEST_NAME,
    MAX_CELLS_PER_CALL,
    MAX_GENES_PER_CELL,
    MIN_DETECTED_GENES,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    MODEL_SUBFOLDER,
    SPECIAL_TOKENS,
    TOKEN_DICTIONARY_FILE,
    VOCAB_SIZE,
    GeneformerPipeline,
    GeneVocabulary,
    load_data_pickle,
    rank_value_encode,
    stage_missing_files,
    validate_inputs,
    verify_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "weights" / MODEL_KEY / MANIFEST_NAME

# 12 genes, so a cell can clear MIN_DETECTED_GENES; medians rise with the index.
_IDS = [f"ENSG{i}" for i in range(1, 13)]
VOCAB = GeneVocabulary(
    tokens={"<pad>": 0, "<mask>": 1, "<cls>": 2, "<eos>": 3, **{gid: 4 + i for i, gid in enumerate(_IDS)}},
    medians={"ENSG1": 1.0, "ENSG2": 2.0, "ENSG3": 4.0, "ENSG4": 0.5, **{gid: 3.0 for gid in _IDS[4:]}},
    symbol_to_id={"GENE1": "ENSG1", "GENE2": "ENSG2"},
)
# Equal raw counts everywhere: the ranking is then decided purely by the corpus medians.
CELL = dict.fromkeys(_IDS, 100.0)


def _wide(value: int = 5) -> dict[str, float]:
    """A cell with just enough detected genes to pass the MIN_DETECTED_GENES rule."""
    return dict.fromkeys(_IDS[:MIN_DETECTED_GENES], float(value))


def _fake(classes=None, logits=None):
    pipe = GeneformerPipeline(lambda rows: [[float(len(r))] * HIDDEN_SIZE for r in rows], "cpu", VOCAB)
    if classes is not None:
        pipe.classes = list(classes)
        rows = logits or [[0.0, 1.0]] * MAX_CELLS_PER_CALL
        pipe._classifier = lambda token_rows: [rows[i] for i in range(len(token_rows))]
    return pipe


def test_identity_constants_and_manifest():
    assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION)
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["modelId"] == MODEL_ID
        assert manifest["revision"] == MODEL_REVISION
        paths = {f["path"] for f in manifest["files"]}
        assert f"{MODEL_SUBFOLDER}/model.safetensors" in paths
        assert {TOKEN_DICTIONARY_FILE, GENE_MEDIAN_DICTIONARY_FILE} <= paths
    config = REPO / "weights" / MODEL_KEY / MODEL_SUBFOLDER / "config.json"
    if config.is_file():
        cfg = json.loads(config.read_text(encoding="utf-8"))
        assert cfg["hidden_size"] == HIDDEN_SIZE
        assert cfg["vocab_size"] == VOCAB_SIZE
        assert cfg["max_position_embeddings"] == MAX_GENES_PER_CELL + 2


def _write_snapshot(tmp_path: Path, content: bytes, sha256: str, revision: str = MODEL_REVISION) -> Path:
    (tmp_path / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [{"path": "config.json", "bytes": len(content), "sha256": sha256}],
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_verify_snapshot_accepts_and_rejects(tmp_path):
    content = b'{"hidden_size": 768}'
    good = hashlib.sha256(content).hexdigest()
    assert verify_snapshot(_write_snapshot(tmp_path, content, good))["revision"] == MODEL_REVISION
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(_write_snapshot(tmp_path, content, "0" * 64))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(_write_snapshot(tmp_path, content, good, revision="0" * 40))
    root = _write_snapshot(tmp_path, content, good)
    (root / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path):
    content = b"{}"
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["files"].append(
        {"path": TOKEN_DICTIONARY_FILE, "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    )
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    fetched = []

    def downloader(rel, dst):
        fetched.append(rel)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        (dst / rel).write_bytes(b"abc")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [TOKEN_DICTIONARY_FILE]
    assert fetched == [TOKEN_DICTIONARY_FILE]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []


def test_stage_refuses_manifest_for_another_model(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"modelId": "other/model", "revision": MODEL_REVISION, "files": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True)


# --- restricted unpickling ---------------------------------------------------------------------


def test_restricted_unpickler_loads_plain_data(tmp_path):
    path = tmp_path / "data.pkl"
    path.write_bytes(pickle.dumps({"ENSG1": 1.5, "ENSG2": 2}))
    assert load_data_pickle(path) == {"ENSG1": 1.5, "ENSG2": 2}


def test_restricted_unpickler_allows_numpy_scalars(tmp_path):
    numpy = pytest.importorskip("numpy")
    path = tmp_path / "medians.pkl"
    path.write_bytes(pickle.dumps({"ENSG1": numpy.float64(1.25)}))
    assert float(load_data_pickle(path)["ENSG1"]) == 1.25


def test_restricted_unpickler_refuses_a_callable_payload(tmp_path):
    import os

    path = tmp_path / "hostile.pkl"
    path.write_bytes(pickle.dumps({"x": os.getcwd}))
    with pytest.raises(pickle.UnpicklingError, match="refusing to resolve"):
        load_data_pickle(path)


def test_restricted_unpickler_refuses_a_non_dict(tmp_path):
    path = tmp_path / "list.pkl"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="expected a dict"):
        load_data_pickle(path)


# --- rank-value encoding -----------------------------------------------------------------------


def test_rank_value_encode_orders_by_median_scaled_expression():
    encoded = rank_value_encode(CELL, VOCAB)
    # Equal raw counts: the genes with the smallest corpus medians rank first.
    assert encoded["ranked_gene_ids"][:3] == ["ENSG4", "ENSG1", "ENSG2"]
    assert encoded["ranked_gene_ids"][-1] == "ENSG3"  # the largest median ranks last
    assert encoded["tokens"] == [VOCAB.tokens[g] for g in encoded["ranked_gene_ids"]]
    assert encoded["n_detected"] == 12 and encoded["n_encoded"] == 12 and encoded["n_truncated"] == 0
    assert encoded["library_size"] == 1200


def test_rank_value_encode_is_scale_invariant():
    scaled = {gene: value * 7 for gene, value in CELL.items()}
    assert rank_value_encode(scaled, VOCAB)["tokens"] == rank_value_encode(CELL, VOCAB)["tokens"]


def test_rank_value_encode_reports_dropped_genes_and_truncation():
    cell = {**CELL, "NOT_A_GENE": 5, "GENE1": 10}
    encoded = rank_value_encode(cell, VOCAB, max_genes=2)
    assert encoded["unknown_genes"] == ["NOT_A_GENE"]
    assert encoded["n_kept"] == 2 and encoded["n_truncated"] == 10
    # A symbol is resolved to its Ensembl id and merged with the id's own count.
    assert encoded["n_encoded"] == 12


def test_rank_value_encode_skips_zero_counts_and_refuses_an_unencodable_cell():
    assert rank_value_encode({**CELL, "ENSG2": 0}, VOCAB)["n_encoded"] == 11
    with pytest.raises(ValueError, match="no gene in this cell could be encoded"):
        rank_value_encode({"NOT_A_GENE": 5}, VOCAB)


# --- input validation --------------------------------------------------------------------------


def test_validate_inputs_manifest_and_rejections():
    manifest = validate_inputs([CELL], VOCAB, names=["c1"])
    assert manifest["verdict"] == "accepted"
    assert manifest["inputs"][0]["id"] == "c1"
    assert manifest["inputs"][0]["tokens_kept"] == 12
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    with pytest.raises(TypeError, match="list of"):
        validate_inputs(CELL, VOCAB)
    with pytest.raises(ValueError, match=f"1..{MAX_CELLS_PER_CALL}"):
        validate_inputs([CELL] * (MAX_CELLS_PER_CALL + 1), VOCAB)
    with pytest.raises(ValueError, match="is empty"):
        validate_inputs([{}], VOCAB)
    with pytest.raises(TypeError, match="must be a number"):
        validate_inputs([{**_wide(), "ENSG1": "5"}], VOCAB)
    with pytest.raises(ValueError, match="finite non-negative"):
        validate_inputs([{**_wide(), "ENSG1": -1}], VOCAB)
    with pytest.raises(ValueError, match=f"at least {MIN_DETECTED_GENES} are required"):
        validate_inputs([{"ENSG1": 5}], VOCAB)
    with pytest.raises(ValueError, match="exactly one id"):
        validate_inputs([CELL], VOCAB, names=["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        validate_inputs([CELL, dict(CELL, ENSG1=1)], VOCAB, names=["a", "a"])


# --- embed / classify / artifact contracts -----------------------------------------------------


def test_embed_contract_with_injected_backend():
    pipe = _fake()
    out = pipe.embed([CELL], names=["c1"])
    assert out["ids"] == ["c1"] and out["dimension"] == HIDDEN_SIZE
    assert len(out["embeddings"][0]) == HIDDEN_SIZE
    assert out["tokens_kept"] == [12] and out["genes_truncated"] == [0]
    assert out["model_revision"] == MODEL_REVISION


def test_embed_rejects_backend_shape_drift():
    pipe = GeneformerPipeline(lambda rows: [[0.0] * 3 for _ in rows], "cpu", VOCAB)
    with pytest.raises(RuntimeError, match="wrong shape"):
        pipe.embed([CELL])


def test_classify_requires_adaptation():
    with pytest.raises(RuntimeError, match="adapt"):
        _fake().classify([CELL])


def test_classify_contract_preserves_class_order():
    pipe = _fake(classes=["programme-A", "programme-B"], logits=[[2.0, 0.0]])
    out = pipe.classify([CELL], names=["c1"])
    prediction = out["predictions"][0]
    assert prediction["label"] == "programme-A"
    assert list(prediction["scores"]) == ["programme-A", "programme-B"]
    assert prediction["scores"]["programme-A"] == pytest.approx(0.8808, abs=1e-3)
    assert prediction["tokens_kept"] == 12
    assert "not calibrated" in out["decision_rule"]


def test_classify_rejects_logits_that_do_not_match_classes():
    pipe = _fake(classes=["a", "b", "c"], logits=[[0.0, 1.0]])
    with pytest.raises(RuntimeError, match="does not match the class list"):
        pipe.classify([CELL])


def test_special_tokens_are_declared():
    assert SPECIAL_TOKENS == ("<pad>", "<mask>", "<cls>", "<eos>")
    assert set(VOCAB.special) == set(SPECIAL_TOKENS)


def test_save_and_load_artifact_require_a_loaded_model(tmp_path):
    pipe = _fake()
    with pytest.raises(RuntimeError, match="adapt"):
        pipe.save_artifact(tmp_path / "adapter")
    with pytest.raises(RuntimeError, match="from_pretrained"):
        pipe.load_artifact(tmp_path / "adapter")
