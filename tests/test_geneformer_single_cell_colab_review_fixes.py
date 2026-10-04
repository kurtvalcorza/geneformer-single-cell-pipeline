"""Regression tests for the geneformer_single_cell_colab review fixes (GF-M1..M2, GF-m1..m4).

They run within CI's install budget (pytest only: no torch, no NumPy, no weights). Adaptation is exercised up to the
point where it would import the model libraries (the `forbid_model_imports` fixture turns that import into an
assertion), evaluation runs on an injected classifier, and the notebook's own Section 4 cell is executed with the
carried package functions, a stand-in gene vocabulary and a stub pipeline. None of this is model or clean-runtime
evidence.
"""
# ruff: noqa: E501  -- test cases quote notebook source lines and refusal messages in full

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import geneformer_single_cell_pipeline as package
from geneformer_single_cell_pipeline import (
    BACKGROUND_GENES,
    HIDDEN_SIZE,
    PROGRAMME_GENES,
    GeneformerPipeline,
    GeneVocabulary,
    generate_sample_dataset,
    load_byod_dataset,
    split_dataset,
    validate_dataset,
    validate_splits,
    write_dataset_csv,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "geneformer_single_cell_colab.ipynb"
PKG = ROOT / "src" / "geneformer_single_cell_pipeline"

# A stand-in vocabulary large enough to build the sample dataset from (as in tests/test_adaptation.py).
N_GENES = 2 * PROGRAMME_GENES + BACKGROUND_GENES + 10
VOCAB = GeneVocabulary(
    tokens={"<pad>": 0, "<mask>": 1, "<cls>": 2, "<eos>": 3, **{f"ENSG{i:05d}": i + 4 for i in range(N_GENES)}},
    medians={f"ENSG{i:05d}": 0.5 + (4.5 * i) / N_GENES for i in range(N_GENES)},
    symbol_to_id={f"SYM{i}": f"ENSG{i:05d}" for i in range(N_GENES)},
)


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _cell(nb: dict, marker: str) -> str:
    found = [_src(c) for c in _code_cells(nb) if marker in _src(c)]
    assert len(found) == 1, marker
    return found[0]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _records(n_per_class: int, classes: tuple[str, ...] = ("programme-A", "programme-B"), seed: int = 11) -> list[dict[str, Any]]:
    """`n_per_class` stand-in cells per class (relabelled sample cells, so every gene is encodable)."""
    cells = generate_sample_dataset(VOCAB, seed=seed, size=2 * n_per_class * len(classes))
    out = []
    for i, cell in enumerate(cells[: n_per_class * len(classes)]):
        out.append({"id": f"cell-{i:03d}", "counts": cell["counts"], "label": classes[i % len(classes)]})
    return out


def _pipe(classes: list[str] | None = None) -> GeneformerPipeline:
    pipe = GeneformerPipeline(lambda rows: [[0.0] * HIDDEN_SIZE for _ in rows], "cpu", VOCAB)
    if classes:
        pipe.classes = list(classes)
        pipe._classifier = lambda rows: [[float(k == 0) for k in range(len(classes))] for _ in rows]
    return pipe


# --- GF-M1: isolated runtime, no in-kernel install ---------------------------------------------------------------


def test_exactly_two_kernel_cells_and_a_hash_locked_isolated_install(nb: dict) -> None:
    kernel = [_src(c) for c in _code_cells(nb) if "# dimer: kernel cell" in _src(c)]
    assert len(kernel) == 2
    install = kernel[0]
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256", 'platform.machine() != "x86_64"'):
        assert needed in install
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in kernel[1]
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    for pin in ("torch==2.14.0", "torchvision==0.29.0", "torchaudio==2.11.0", "transformers==4.57.6", "numpy==2.5.3", "safetensors==0.8.0", "huggingface-hub==0.36.2"):
        assert pin in lock
    assert "geneformer-single-cell-pipeline (pyproject.toml)" in lock and "esm2" not in lock and "ast-audio" not in lock


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(nb: dict, monkeypatch: pytest.MonkeyPatch, real_google: bool) -> None:
    """find_spec("google.colab") (accelerate does this) must not raise on the worker's stubs (fleet Colab failure)."""
    namespace: dict[str, Any] = {"SKIP_INSTALL": True, "__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_src(_code_cells(nb)[1]), "router", "exec"), namespace)
    worker = namespace["_WORKER_SOURCE"]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    for name in ("google", "google.colab", "google.colab.files"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    shim_globals = {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None}
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), shim_globals)
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google.colab", "google.colab.files"):
            sys.modules.pop(name, None)


def test_release_record_no_longer_counts_the_restarted_run_as_a_pass() -> None:
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "after the expected fresh-process restart" not in text
    assert "One expected fresh-process restart followed the install cell" not in text
    assert text.count("**Passed only after a manual restart** — not a one-pass Run all, not promotion evidence.") == 2
    assert "`restarted: false`" in text
    assert "Current status: **Candidate" in (ROOT / "STATUS.md").read_text(encoding="utf-8")
    for name in ("README.md", "STATUS.md", "tutorials/README.md"):
        assert "Release-grade** —" not in (ROOT / name).read_text(encoding="utf-8"), name
        assert "**Release-grade.**" not in (ROOT / name).read_text(encoding="utf-8"), name


# --- GF-m1: one BYOD size contract; no stale head after a dataset change -------------------------------------------


def test_stated_minimum_dataset_meets_every_split_minimum(forbid_model_imports, tmp_path: Path) -> None:
    records = _records(4)  # 8 records, 4 per class: the stated minimum (>= 8 records, >= 3 per class)
    manifest = validate_dataset(records, vocabulary=VOCAB)
    splits = split_dataset(records, seed=42)
    assert {k: len(v) for k, v in splits.items()} == {"train": 4, "validation": 2, "test": 2}
    manifests = validate_splits(splits, manifest["classes"], vocabulary=VOCAB)
    assert all(m["classes"] == ["programme-A", "programme-B"] for m in manifests.values())
    pipe = _pipe()
    pipe.model, pipe.weights_dir = object(), tmp_path
    # Validation passes, so adapt reaches the model import (which the fixture turns into an assertion).
    with pytest.raises(AssertionError, match="model dependency imported"):
        pipe.adapt(splits["train"], splits["validation"], classes=manifest["classes"])
    # Three classes of three records (9 >= 8) also reach adaptation: one record per class per split.
    three = _records(3, ("a", "b", "c"))
    three_splits = split_dataset(three, seed=42)
    assert {k: len(v) for k, v in three_splits.items()} == {"train": 3, "validation": 3, "test": 3}
    validate_splits(three_splits, vocabulary=VOCAB)
    with pytest.raises(AssertionError, match="model dependency imported"):
        pipe.adapt(three_splits["train"], three_splits["validation"])


def test_split_refusals_name_the_split(forbid_model_imports, tmp_path: Path) -> None:
    splits = split_dataset(_records(6, ("a", "b", "c")), seed=42)
    no_c = [r for r in splits["validation"] if r["label"] != "c"]
    with pytest.raises(ValueError, match=r"^validation split: classes \['c'\] have fewer than 1"):
        validate_splits({**splits, "validation": no_c})
    pipe = _pipe()
    pipe.model, pipe.weights_dir = object(), tmp_path
    with pytest.raises(ValueError, match=r"^validation split: "):
        pipe.adapt(splits["train"], no_c)
    with pytest.raises(ValueError, match=r"^train split: "):
        pipe.adapt(splits["train"][:1], splits["validation"])


def test_split_message_names_the_size_a_class_needs() -> None:
    records = _records(6)  # 6 per class: 3 validation + 3 test leave no training record
    with pytest.raises(ValueError, match=r"class 'programme-A' has 6 records; .* it needs at least 7 to leave one per split"):
        split_dataset(records, val_fraction=0.45, test_fraction=0.45)


def test_evaluate_accepts_a_small_split_and_refuses_without_a_head(tmp_path: Path) -> None:
    splits = split_dataset(_records(4), seed=42)
    pipe = _pipe(["programme-A", "programme-B"])
    metrics = pipe.evaluate(splits["test"])  # 2 records: used to fail the 8-record dataset minimum
    assert metrics["n"] == 2 and metrics["accuracy"] == 0.5
    pipe.reset_adaptation()
    assert pipe.classes == [] and pipe._classifier is None and pipe.adaptation == {}
    with pytest.raises(RuntimeError, match="requires an adapted head"):
        pipe.evaluate(splits["test"])
    with pytest.raises(RuntimeError, match="requires an adapted head"):
        pipe.save_artifact(tmp_path / "never-written")
    assert not (tmp_path / "never-written").exists()


def _section4_namespace(pipe: Any) -> dict[str, Any]:
    ns = {name: getattr(package, name) for name in package.__all__}
    ns.update({"pipe": pipe, "__name__": "__main__"})
    return ns


def _run_section4(nb: dict, ns: dict, *, use_byod: bool, byod_path: str = "") -> str:
    src = _cell(nb, "USE_BYOD = False  # @param")
    if use_byod:
        src = src.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param")
        src = src.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {byod_path!r}  # @param")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(src, "section4", "exec"), ns)
    return out.getvalue()


def test_section4_byod_path_runs_outside_colab_keeps_the_template_and_drops_the_old_head(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _run_section4(nb, _section4_namespace(_pipe(["x", "y"])), use_byod=False)
    template = tmp_path / "outputs" / "geneformer_single_cell_sample_dataset.csv"
    template_digest = hashlib.sha256(template.read_bytes()).hexdigest()
    pipe = _pipe(["programme-A", "programme-B"])
    pipe.adaptation = {"trainable_layers": 2}
    path = tmp_path / "my_cells.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _records(4)) + "\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    ns = _section4_namespace(pipe)
    printed = _run_section4(nb, ns, use_byod=True, byod_path=str(path))
    assert "google.colab" not in sys.modules
    assert ns["data_source"] == "BYOD (my_cells.jsonl)"
    assert ns["dataset_csv"] == "outputs/geneformer_single_cell_byod_dataset.csv"
    assert (tmp_path / "outputs" / "geneformer_single_cell_byod_dataset.csv").is_file()
    assert hashlib.sha256(template.read_bytes()).hexdigest() == template_digest, "BYOD must not overwrite the sample template"
    assert [len(ns[k]) for k in ("train_records", "val_records", "test_records")] == [4, 2, 2]
    assert pipe.classes == [] and pipe._classifier is None, "the sample head must be dropped when the dataset changes"
    assert "'validation': {'n': 2" in printed


def test_section4_default_path_writes_the_sample_template(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns = _section4_namespace(_pipe(["a", "b"]))
    _run_section4(nb, ns, use_byod=False)
    assert ns["dataset_csv"] == "outputs/geneformer_single_cell_sample_dataset.csv"
    assert [len(ns[k]) for k in ("train_records", "val_records", "test_records")] == [36, 12, 16]
    assert not (tmp_path / "outputs" / "geneformer_single_cell_byod_dataset.csv").exists()


def _fake_colab(monkeypatch: pytest.MonkeyPatch, replies: list[dict]) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab = types.ModuleType("google.colab")
    colab.__path__ = []
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: replies.pop(0)
    colab.files = files
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def test_byod_intake_errors_are_actionable(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Outside Colab with no BYOD_PATH: a named instruction, not a bare ImportError.
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(RuntimeError, match=r"upload dialog exists only on Google Colab\. Set BYOD_PATH"):
        _run_section4(nb, _section4_namespace(_pipe()), use_byod=True)
    # A cancelled upload: a named condition, not a bare StopIteration.
    _fake_colab(monkeypatch, [{}])
    with pytest.raises(ValueError, match=r"Upload exactly one \.csv, \.json or \.jsonl file \(got 0"):
        _run_section4(nb, _section4_namespace(_pipe()), use_byod=True)
    # An .h5ad file is refused by its suffix before decoding (it used to raise UnicodeDecodeError).
    h5ad = tmp_path / "cells.h5ad"
    h5ad.write_bytes(b"\x89HDF\r\n\x1a\n" + bytes(range(256)))
    with pytest.raises(ValueError, match=r"unsupported BYOD file type '\.h5ad' \(cells\.h5ad\); use \.csv, \.json or \.jsonl\. .*genes-as-columns CSV"):
        load_byod_dataset(h5ad)
    # A UTF-16 CSV: a named encoding problem and the fix.
    write_dataset_csv(_records(4), tmp_path / "plain.csv")
    utf16 = tmp_path / "export.csv"
    utf16.write_bytes((tmp_path / "plain.csv").read_text(encoding="utf-8").encode("utf-16"))
    with pytest.raises(ValueError, match=r"export\.csv is not UTF-8 text .*save it as UTF-8"):
        load_byod_dataset(utf16)


# --- GF-M2 / GF-m3 / GF-m4: guided layer, infrastructure, activity, embedding reading guide --------------------------


def test_guided_layer_and_infrastructure_cells(nb: dict) -> None:
    md = _markdown(nb)
    for marker, least in (
        ("**Who this is for.**", 1),
        ("**Input → Model → Output.**", 1),
        ("**How to use this notebook.**", 1),
        ("**Roadmap:**", 1),
        ("**Predict before running:**", 9),
        ("**What to notice:**", 9),
        ("<summary>Check your reasoning</summary>", 10),
        ("## Troubleshooting", 1),
        ("## Glossary", 1),
        ("## Conclusion (your notes)", 1),
        ("> **Infrastructure.**", 3),
    ):
        assert md.count(marker) >= least, marker
    titled = [c for c in _code_cells(nb) if _src(c).startswith("# @title Infrastructure:")]
    assert len(titled) == 7 and all(c["metadata"].get("cellView") == "form" for c in titled)
    carried = [c for c in _code_cells(nb) if c["metadata"].get("dimer", {}).get("embedded_module")]
    assert len(carried) == 3 and all(c["metadata"].get("jupyter", {}).get("source_hidden") for c in carried)
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"


def test_activity_names_the_rerun_scope_and_exports_describe_one_model(nb: dict) -> None:
    md = _markdown(nb)
    assert "## 13. Your turn — change one thing: train for fewer epochs" in md
    assert "**Predict → Change one thing → Run → Observe → Explain.**" in md
    assert "select the Section 8 cell and choose **Runtime → Run after**" in md
    assert "Optional experiments (do not affect the default path)" not in md
    for item in ("**Fewer epochs:**", "**Head only:**", "**Another split:**", "**Your own labelled cells:**"):
        line = next(x for x in md.splitlines() if item in x)
        assert "**Run after** from Section" in line, item
    assert "run_history.append(" in _cell(nb, "val_metrics = pipe.evaluate(val_records)")
    result = _cell(nb, "result_payload = {")
    assert "assert evaluation_report['adaptation']['epochs'] == artifact_manifest['adaptation']['epochs'] == EPOCHS" in result
    assert "for row in run_history:" in _cell(nb, "columns = ['run', 'data'")


def test_embedding_section_explains_the_two_cosines_without_hard_coding_them(nb: dict) -> None:
    md = _markdown(nb)
    notice = next(block for block in md.split("**What to notice:**")[1:] if "768 dimensions" in block)
    head = notice.split("<details>")[0]
    assert "read the gap between them, not their size" in head
    assert not re.search(r"0\.99\d", head), "the What to notice note must not hard-code the cosine values"
    assert "mean pooling" in notice and "separability is measured with labels" in notice


# --- GF-m2: template artefact, stale procedure value, digest labels ------------------------------------------------


def test_prerequisites_braces_procedure_defaults_and_digest_labels(nb: dict) -> None:
    md = _markdown(nb)
    assert "{{" not in md and "}}" not in md
    assert "records are `{id, counts, label}`" in md
    procedure = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    form = dict(re.findall(r"^(\w+) = (.+?)  # @param", "\n".join(_src(c) for c in _code_cells(nb)), re.M))
    for name in ("EPOCHS", "LEARNING_RATE", "BATCH_SIZE", "TRAINABLE_LAYERS", "VAL_FRACTION", "TEST_FRACTION", "SEED"):
        assert f"`{name} = {form[name]}`" in procedure, name
    runtime = _cell(nb, "NOTEBOOK_SOURCE = {")
    assert "'module_sha256'" not in runtime and "'modules_sha256_concat'" in runtime
    per_module = json.loads(re.search(r"'per_module_sha256': (\{[^}]*\})", runtime).group(1).replace("'", '"'))
    for rel, digest in per_module.items():
        assert hashlib.sha256((ROOT / rel).read_text(encoding="utf-8").encode("utf-8")).hexdigest() == digest, rel
    assert set(per_module) == {f"src/geneformer_single_cell_pipeline/{m}" for m in ("pipeline.py", "samples.py", "metrics.py")}
    generated = nb["metadata"]["dimer"]["generated_from"]
    assert "module_sha256" not in generated and generated["per_module_sha256"] == per_module


def test_release_procedure_has_a_byod_gate() -> None:
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "the **BYOD gate (REL12)**" in text
    assert "record one refused input" in text
