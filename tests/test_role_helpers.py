"""Offline tests for the public validation-stage helpers (input manifest and dataset manifest)."""

from __future__ import annotations

from geneformer_single_cell_pipeline import (
    INPUT_SCHEMA,
    MAX_CELLS_PER_CALL,
    MAX_GENES_PER_CELL,
    MIN_DETECTED_GENES,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SUBFOLDER,
    GeneVocabulary,
    validate_inputs,
)

VOCAB = GeneVocabulary(
    tokens={"<pad>": 0, "<mask>": 1, "<cls>": 2, "<eos>": 3, **{f"ENSG{i}": i + 4 for i in range(12)}},
    medians={f"ENSG{i}": 1.0 + i for i in range(12)},
    symbol_to_id={"SYM0": "ENSG0"},
)
CELL = {f"ENSG{i}": 5.0 + i for i in range(12)}


def test_validate_inputs_returns_manifest_with_schema_and_identity():
    manifest = validate_inputs([CELL], VOCAB, names=["c1"])
    assert manifest["verdict"] == "accepted" and manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["cells"] == [1, MAX_CELLS_PER_CALL]
    assert manifest["schema"]["detected_genes_per_cell"] == [MIN_DETECTED_GENES, MAX_GENES_PER_CELL]
    row = manifest["inputs"][0]
    assert row["id"] == "c1" and row["detected_genes"] == 12 and row["tokens_kept"] == 12
    assert row["unknown_genes"] == 0 and row["genes_without_median"] == 0
    assert manifest["n_cells"] == 1 and manifest["max_tokens_observed"] == 12
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert manifest["model_subfolder"] == MODEL_SUBFOLDER


def test_validate_inputs_default_ids_and_symbol_resolution():
    manifest = validate_inputs([{**CELL, "SYM0": 3.0}], VOCAB)
    assert [row["id"] for row in manifest["inputs"]] == ["cell-0"]
    # The symbol resolves onto ENSG0, so it is counted as detected but not as unknown.
    assert manifest["inputs"][0]["unknown_genes"] == 0
