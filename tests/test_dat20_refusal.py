"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

import geneformer_single_cell_pipeline as package
from geneformer_single_cell_pipeline import (
    BACKGROUND_GENES,
    HIDDEN_SIZE,
    PROGRAMME_GENES,
    GeneformerPipeline,
    GeneVocabulary,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/geneformer_single_cell_colab.ipynb"
BYOD_CELL = "geneformer_single_cell-17"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


STEM = "geneformer_single_cell"
EARLIER = {"records": "old", "splits": "old", "test_records": "old", "embedding_result": "old", "adapt_result": "old", "evaluation_report": "old", "result_payload": "old", "CLASSES": "old"}
SCRIPTED = "/content/sample_data/california_housing_test.csv"
N_GENES = 2 * PROGRAMME_GENES + BACKGROUND_GENES + 10
VOCAB = GeneVocabulary(
    tokens={"<pad>": 0, "<mask>": 1, "<cls>": 2, "<eos>": 3, **{f"ENSG{i:05d}": i + 4 for i in range(N_GENES)}},
    medians={f"ENSG{i:05d}": 0.5 + (4.5 * i) / N_GENES for i in range(N_GENES)},
    symbol_to_id={f"SYM{i}": f"ENSG{i:05d}" for i in range(N_GENES)},
)


def _new_pipe() -> GeneformerPipeline:
    return GeneformerPipeline(lambda rows: [[0.0] * HIDDEN_SIZE for _ in rows], "cpu", VOCAB)


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    (out / f"{STEM}_adapter").mkdir(parents=True)
    for name in ("sample_dataset.csv", "embeddings.csv", "evaluation_report.json", "result.json", "predictions.csv"):
        (out / f"{STEM}_{name}").write_text("x", encoding="utf-8")


def _section4(monkeypatch, tmp_path, path: str, *, use_byod: bool = True, source: str | None = None, extra: dict | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    _seed_outputs(tmp_path)
    source = source or _cell(BYOD_CELL)
    source = source[: source.index("train_records, val_records, test_records = splits")]
    if use_byod:
        source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
    source = source.replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)
    ns = {name: getattr(package, name) for name in package.__all__}
    ns.update(EARLIER, pipe=_new_pipe(), **(extra or {}))
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "california_housing_test.csv").write_text("longitude,latitude,median_house_value\n-122.05,37.37,344700.0\n", encoding="utf-8")
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "object.json").write_text('{"id": "a"}', encoding="utf-8")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    (tmp_path / "text_count.csv").write_text("id,ENSG00000,label\nc1,many,a\n", encoding="utf-8")
    return {
        "wrong columns (the scripted sample_data CSV)": ("california_housing_test.csv", "CSV must start with an 'id' column"),
        "not a dataset suffix": ("hostname", "unsupported BYOD file type"),
        "invalid JSON": ("bad.json", "not valid JSON"),
        "JSON that is not an array (TypeError)": ("object.json", "TypeError: JSON dataset must be a top-level array"),
        "empty file": ("empty.csv", "is empty"),
        "a folder": ("folder", "does not exist or is not a file"),
        "missing": ("nope.csv", "does not exist or is not a file"),
        "a count that is not a number": ("text_count.csv", "is not a number"),
    }


CASES = ["wrong columns (the scripted sample_data CSV)", "not a dataset suffix", "invalid JSON", "JSON that is not an array (TypeError)", "empty file", "a folder", "missing", "a count that is not a number"]


@pytest.mark.parametrize("case", CASES)
def test_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    file_name, rule = _cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / file_name))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


def test_refusal_clears_the_session(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.chdir(tmp_path)
    source = _cell(BYOD_CELL)
    source = source[: source.index("train_records, val_records, test_records = splits")].replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", "BYOD_PATH = 'nope.csv'", 1)
    ns = {name: getattr(package, name) for name in package.__all__}
    pipe = _new_pipe()
    ns.update(EARLIER, pipe=pipe)
    with pytest.raises(ValueError, match="^nope.csv: BYOD_PATH 'nope.csv' does not exist"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] is pipe and ns["vocabulary"] is pipe.vocabulary


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith("Refused: (no file): the upload dialog exists only on Google Colab")


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "got 0"), ({"a.csv": b"1", "b.csv": b"2"}, "a.csv, b.csv", "got 2")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: upload exactly one") and rule in line


def test_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "ok.csv").write_text("id,ENSG00000,label\n", encoding="utf-8")
    source = _cell(BYOD_CELL).replace("        records = load_byod_dataset(byod_path)\n", "        raise RuntimeError('internal bug')\n", 1)
    assert "internal bug" in source
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "ok.csv"), source=source)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is removed for an internal error


def test_sample_path_errors_are_not_refusals(monkeypatch, tmp_path):
    def broken(*args, **kwargs):
        raise ValueError("sample generator bug")

    with pytest.raises(ValueError, match="sample generator bug") as caught:
        _section4(monkeypatch, tmp_path, "", use_byod=False, extra={"generate_sample_dataset": broken})
    assert not getattr(caught.value, "dimer_refusal", False)


def test_default_path_still_splits(monkeypatch, tmp_path):
    ns = _section4(monkeypatch, tmp_path, "", use_byod=False)
    assert set(ns["splits"]) == {"train", "validation", "test"} and ns["records"] != "old"


def test_valid_byod_file_still_loads(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    package.write_dataset_csv(package.generate_sample_dataset(VOCAB), tmp_path / "good.csv")
    ns = _section4(monkeypatch, tmp_path, str(tmp_path / "good.csv"))
    assert ns["data_source"] == "BYOD (good.csv)" and set(ns["splits"]) == {"train", "validation", "test"}


@pytest.mark.skipif(not Path(SCRIPTED).is_file(), reason="needs Colab's /content/sample_data, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_sample_data_csv(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, SCRIPTED)
    assert _refusal_line(caught.value).startswith("Refused: california_housing_test.csv: ValueError: CSV must start with an 'id' column")
