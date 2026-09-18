"""Geneformer V2-104M (`ctheodoris/Geneformer`) DIMER pipeline: verified snapshot, rank-value
encoding of single-cell transcriptomes, cell embeddings, and bounded cell-state classification
fine-tuning with a portable adapter artifact.

Everything model-related is imported lazily so that snapshot verification and input validation run
(and can refuse) before `torch` or `transformers` are imported (fleet RTM-001).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "ctheodoris/Geneformer"
MODEL_REVISION = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "geneformer-v2-104m"
# The repository hosts several checkpoints; this pipeline packages exactly one of them.
MODEL_SUBFOLDER = "Geneformer-V2-104M"
DICTIONARY_DIR = "geneformer"
TOKEN_DICTIONARY_FILE = "geneformer/token_dictionary_gc104M.pkl"
GENE_MEDIAN_DICTIONARY_FILE = "geneformer/gene_median_dictionary_gc104M.pkl"
GENE_NAME_DICTIONARY_FILE = "geneformer/gene_name_id_dict_gc104M.pkl"
ARTIFACT_FORMAT = "org.valcorza.geneformer-single-cell.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Ceilings. The V2 checkpoint's config.json declares max_position_embeddings 4096 and the upstream
# README states an input size of 4096 tokens for V2; `<cls>` and `<eos>` take two of them, so 4094
# ranked genes is the most a cell can carry. Longer rank lists are truncated at the tail (the
# lowest-ranked genes), which is the upstream behaviour, and the pipeline reports the truncation.
MAX_INPUT_TOKENS = 4096
MAX_GENES_PER_CELL = MAX_INPUT_TOKENS - 2
MAX_CELLS_PER_CALL = 32
HIDDEN_SIZE = 768  # config.json hidden_size; the width of every `embed` row
VOCAB_SIZE = 20275  # config.json vocab_size; equals len(token_dictionary)
SPECIAL_TOKENS = ("<pad>", "<mask>", "<cls>", "<eos>")
MIN_DETECTED_GENES = 10  # a cell with fewer measured genes cannot produce a meaningful ranking

# The only globals the pinned data pickles legitimately contain: the gene-median dictionary stores
# numpy float64 scalars. Everything else is refused, so loading these files parses data rather than
# executing arbitrary code (see docs/WEIGHTS.md, "Dictionary trust boundary").
ALLOWED_PICKLE_GLOBALS = frozenset(
    {
        # numpy 1.x and numpy 2.x spell the scalar reconstructor differently; both appear in the
        # wild for the same pinned bytes depending on the numpy that reads them.
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "dtype"),
    }
)


def _is_allowed_global(module: str, name: str) -> bool:
    """The allowlist plus numpy 2's concrete dtype classes (`numpy.dtypes.Float64DType`)."""
    if (module, name) in ALLOWED_PICKLE_GLOBALS:
        return True
    return module == "numpy.dtypes" and name.endswith("DType")


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that refuses every global except the allowlist above."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: D102
        if not _is_allowed_global(module, name):
            raise pickle.UnpicklingError(
                f"refusing to resolve {module}.{name} while loading a pinned data dictionary; "
                f"only {sorted(ALLOWED_PICKLE_GLOBALS)} are allowed"
            )
        return super().find_class(module, name)


def load_data_pickle(path: str | Path) -> dict[Any, Any]:
    """Load one digest-verified data dictionary through `RestrictedUnpickler`."""
    with open(path, "rb") as fh:
        obj = RestrictedUnpickler(fh).load()
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a dict, got {type(obj).__name__}")
    return obj


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest.hexdigest()} != manifest {entry['sha256']}")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the Geneformer snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _verify_manifest(root, MODEL_ID, MODEL_REVISION)


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


@dataclass(frozen=True)
class GeneVocabulary:
    """The pinned gene dictionaries: Ensembl id -> token, Ensembl id -> corpus median, symbol -> id."""

    tokens: dict[str, int]
    medians: dict[str, float]
    symbol_to_id: dict[str, str]

    @property
    def special(self) -> dict[str, int]:
        return {name: self.tokens[name] for name in SPECIAL_TOKENS if name in self.tokens}

    @property
    def gene_ids(self) -> list[str]:
        """Ensembl ids that have both a token and a corpus median (the encodable vocabulary)."""
        return sorted(gid for gid in self.tokens if gid in self.medians)

    def resolve(self, name: str) -> str | None:
        """Map a gene symbol to its Ensembl id; an Ensembl id passes through unchanged."""
        if name in self.tokens:
            return name
        return self.symbol_to_id.get(name)


def load_gene_vocabulary(path: str | Path | None = None) -> GeneVocabulary:
    """Load the three pinned dictionaries through the restricted unpickler and sanity-check them."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    tokens = {str(k): int(v) for k, v in load_data_pickle(root / TOKEN_DICTIONARY_FILE).items()}
    medians = {str(k): float(v) for k, v in load_data_pickle(root / GENE_MEDIAN_DICTIONARY_FILE).items()}
    symbols = {str(k): str(v) for k, v in load_data_pickle(root / GENE_NAME_DICTIONARY_FILE).items()}
    if len(tokens) != VOCAB_SIZE:
        raise ValueError(f"token dictionary holds {len(tokens)} entries, config declares {VOCAB_SIZE}")
    missing_special = [name for name in SPECIAL_TOKENS if name not in tokens]
    if missing_special:
        raise ValueError(f"token dictionary is missing special tokens {missing_special}")
    return GeneVocabulary(tokens, medians, symbols)


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "1..MAX_CELLS_PER_CALL cells, each a mapping of Ensembl gene id (or gene symbol) "
        "to a non-negative count"
    ),
    "cells": [1, MAX_CELLS_PER_CALL],
    "detected_genes_per_cell": [MIN_DETECTED_GENES, MAX_GENES_PER_CELL],
    "input_tokens": MAX_INPUT_TOKENS,
    "preprocessing": (
        "counts are normalised to a fixed library size, divided by the gene's corpus median from the "
        "pinned gene_median dictionary, ranked in descending order, truncated to the input size, and "
        "mapped to gene tokens between <cls> and <eos> (Geneformer rank-value encoding)"
    ),
}


def _check_cells(cells: Any, names: Any = None) -> tuple[list[dict[str, float]], list[str]]:
    """Raise TypeError/ValueError naming the first violated rule; return (cells, ids).

    ``embed``, ``classify`` and ``validate_inputs`` all route through this function so their
    acceptance criteria cannot diverge.
    """
    if isinstance(cells, Mapping) or not isinstance(cells, Sequence) or isinstance(cells, str | bytes):
        raise TypeError("cells must be a list of {gene: count} mappings (one mapping per cell)")
    if not 1 <= len(cells) <= MAX_CELLS_PER_CALL:
        raise ValueError(f"cells must hold 1..{MAX_CELLS_PER_CALL} items, got {len(cells)}")
    checked: list[dict[str, float]] = []
    for i, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise TypeError(f"cells[{i}] must be a mapping of gene to count, got {type(cell).__name__}")
        if not cell:
            raise ValueError(f"cells[{i}] is empty")
        counts: dict[str, float] = {}
        for gene, value in cell.items():
            if not isinstance(gene, str) or not gene.strip():
                raise TypeError(f"cells[{i}] has a non-string gene key {gene!r}")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"cells[{i}][{gene!r}] must be a number, got {type(value).__name__}")
            if value < 0 or value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"cells[{i}][{gene!r}] must be a finite non-negative count, got {value!r}")
            counts[gene] = float(value)
        detected = sum(1 for v in counts.values() if v > 0)
        if detected < MIN_DETECTED_GENES:
            raise ValueError(
                f"cells[{i}] has {detected} detected gene(s); at least {MIN_DETECTED_GENES} are required "
                "to rank a transcriptome"
            )
        checked.append(counts)
    if names is None:
        ids = [f"cell-{i}" for i in range(len(checked))]
    else:
        if isinstance(names, str | bytes) or not isinstance(names, Sequence) or len(names) != len(checked):
            raise ValueError("names must be a list with exactly one id per cell")
        ids = [str(n) for n in names]
        if len(set(ids)) != len(ids):
            raise ValueError("names must be unique")
    return checked, ids


def rank_value_encode(
    counts: Mapping[str, float],
    vocabulary: GeneVocabulary,
    *,
    target_sum: float = 10_000.0,
    max_genes: int = MAX_GENES_PER_CELL,
) -> dict[str, Any]:
    """Geneformer rank-value encoding of one cell; returns the token ids and what was dropped.

    Genes are normalised to `target_sum` total counts, divided by their corpus median expression,
    ranked in descending order and truncated to `max_genes`. Genes with a zero count, genes absent
    from the token dictionary, and genes without a corpus median are dropped and reported rather
    than silently ignored (VAL7).
    """
    unknown: list[str] = []
    no_median: list[str] = []
    resolved: dict[str, float] = {}
    for gene, value in counts.items():
        if value <= 0:
            continue
        gene_id = vocabulary.resolve(gene)
        if gene_id is None:
            unknown.append(gene)
            continue
        if gene_id not in vocabulary.medians:
            no_median.append(gene_id)
            continue
        resolved[gene_id] = resolved.get(gene_id, 0.0) + value
    if not resolved:
        raise ValueError(
            "no gene in this cell could be encoded: none of the detected genes carry both a Geneformer "
            "token and a corpus median (check that gene identifiers are human Ensembl ids or symbols)"
        )
    total = sum(resolved.values())
    scaled = {gid: (value / total) * target_sum / vocabulary.medians[gid] for gid, value in resolved.items()}
    ranked = sorted(scaled.items(), key=lambda item: (-item[1], item[0]))
    kept = ranked[:max_genes]
    return {
        "tokens": [vocabulary.tokens[gid] for gid, _ in kept],
        "ranked_gene_ids": [gid for gid, _ in kept],
        "n_detected": sum(1 for v in counts.values() if v > 0),
        "n_encoded": len(resolved),
        "n_kept": len(kept),
        "n_truncated": max(0, len(ranked) - len(kept)),
        "unknown_genes": sorted(set(unknown)),
        "genes_without_median": sorted(set(no_median)),
        "library_size": total,
    }


def validate_inputs(
    cells: Sequence[Mapping[str, float]],
    vocabulary: GeneVocabulary,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: encode every cell and return the input manifest (schema, observations, verdict).

    Rejection is reported by raising exactly as ``embed``/``classify`` would.
    """
    checked, ids = _check_cells(cells, names)
    rows = []
    for cid, counts in zip(ids, checked, strict=True):
        encoded = rank_value_encode(counts, vocabulary)
        rows.append(
            {
                "id": cid,
                "detected_genes": encoded["n_detected"],
                "encoded_genes": encoded["n_encoded"],
                "tokens_kept": encoded["n_kept"],
                "genes_truncated": encoded["n_truncated"],
                "unknown_genes": len(encoded["unknown_genes"]),
                "genes_without_median": len(encoded["genes_without_median"]),
                "library_size": encoded["library_size"],
            }
        )
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": rows,
        "n_cells": len(checked),
        "max_tokens_observed": max(row["tokens_kept"] for row in rows),
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_subfolder": MODEL_SUBFOLDER,
    }


def _softmax(logits: Sequence[float]) -> list[float]:
    import math

    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


@dataclass
class GeneformerPipeline:
    """Geneformer V2-104M pipeline: `embed` always; `classify` after `adapt` or `from_artifact`."""

    _embedder: Callable[[list[list[int]]], list[list[float]]]
    device: str
    vocabulary: GeneVocabulary
    load_warnings: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    _classifier: Callable[[list[list[int]]], list[list[float]]] | None = None
    model: Any = None
    classifier_model: Any = None
    weights_dir: Path | None = None
    adaptation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> GeneformerPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(f"no snapshot manifest at {root} and allow_download={allow_download}")
        # Stage, verify and load the data dictionaries before importing model libraries (RTM-001).
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        vocabulary = load_gene_vocabulary(root)
        import torch
        from transformers import BertModel

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = BertModel.from_pretrained(
                str(root / MODEL_SUBFOLDER),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                add_pooling_layer=False,
            )
        model = model.to(resolved_device).eval()
        messages = [f"{w.category.__name__}: {w.message}" for w in caught]
        pipe = cls(
            cls._make_embedder(model, resolved_device, vocabulary),
            resolved_device,
            vocabulary,
            messages,
        )
        pipe.model, pipe.weights_dir = model, root
        return pipe

    # -- backends ---------------------------------------------------------------------------------

    @staticmethod
    def _batch(token_rows: list[list[int]], vocabulary: GeneVocabulary, device: str) -> dict[str, Any]:
        """Pad rank-value rows into `<cls> … <eos>` batches with an attention mask."""
        import torch

        cls_id, eos_id, pad_id = (vocabulary.tokens[t] for t in ("<cls>", "<eos>", "<pad>"))
        rows = [[cls_id, *row, eos_id] for row in token_rows]
        width = max(len(row) for row in rows)
        input_ids = [row + [pad_id] * (width - len(row)) for row in rows]
        attention = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        }

    @classmethod
    def _make_embedder(
        cls, model: Any, device: str, vocabulary: GeneVocabulary
    ) -> Callable[[list[list[int]]], list[list[float]]]:
        import torch

        def embedder(token_rows: list[list[int]]) -> list[list[float]]:
            batch = cls._batch(token_rows, vocabulary, device)
            # no_grad, not inference_mode: tensors produced here must stay usable by a later
            # training epoch that shares this module.
            with torch.no_grad():
                hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return pooled.float().cpu().tolist()

        return embedder

    @classmethod
    def _make_classifier(
        cls, model: Any, device: str, vocabulary: GeneVocabulary
    ) -> Callable[[list[list[int]]], list[list[float]]]:
        import torch

        def classifier(token_rows: list[list[int]]) -> list[list[float]]:
            batch = cls._batch(token_rows, vocabulary, device)
            with torch.no_grad():
                logits = model(**batch).logits
            return logits.float().cpu().tolist()

        return classifier

    def encode(self, cells: Sequence[Mapping[str, float]]) -> list[dict[str, Any]]:
        """Rank-value encode every cell with this pipeline's pinned vocabulary."""
        return [rank_value_encode(cell, self.vocabulary) for cell in cells]

    # -- public stages ----------------------------------------------------------------------------

    def embed(
        self, cells: Sequence[Mapping[str, float]], *, names: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Mean-pooled last-hidden-state cell embedding (HIDDEN_SIZE floats per cell)."""
        checked, ids = _check_cells(cells, names)
        encoded = self.encode(checked)
        vectors = self._embedder([e["tokens"] for e in encoded])
        if len(vectors) != len(checked) or any(len(v) != HIDDEN_SIZE for v in vectors):
            raise RuntimeError("backend returned embeddings of the wrong shape")
        return {
            "ids": ids,
            "embeddings": [[float(x) for x in v] for v in vectors],
            "dimension": HIDDEN_SIZE,
            "pooling": "mean of the last hidden state over the cell's tokens (padding excluded)",
            "unit": "one vector per cell; representations, not predictions",
            "tokens_kept": [e["n_kept"] for e in encoded],
            "genes_truncated": [e["n_truncated"] for e in encoded],
            "n_cells": len(checked),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def classify(
        self, cells: Sequence[Mapping[str, float]], *, names: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Cell-state scores and argmax label; requires a prior `adapt` or `from_artifact`."""
        if self._classifier is None or not self.classes:
            raise RuntimeError(
                "classify requires an adapted head: call adapt(...) or load from_artifact(...) first"
            )
        checked, ids = _check_cells(cells, names)
        encoded = self.encode(checked)
        logits = self._classifier([e["tokens"] for e in encoded])
        predictions = []
        for cid, enc, row in zip(ids, encoded, logits, strict=True):
            if len(row) != len(self.classes):
                raise RuntimeError("backend returned a logits row that does not match the class list")
            scores = _softmax(row)
            best = max(range(len(scores)), key=scores.__getitem__)
            predictions.append(
                {
                    "id": cid,
                    "tokens_kept": enc["n_kept"],
                    "label": self.classes[best],
                    "score": scores[best],
                    "scores": dict(zip(self.classes, scores, strict=True)),
                }
            )
        return {
            "predictions": predictions,
            "classes": list(self.classes),
            "decision_rule": (
                "argmax over softmax(logits); scores are softmax outputs, not calibrated probabilities"
            ),
            "n_cells": len(checked),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adaptation": dict(self.adaptation),
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Held-out cell-state classification metrics (see metrics.classification_metrics)."""
        from .metrics import classification_metrics
        from .samples import validate_dataset

        validate_dataset(records, vocabulary=self.vocabulary, classes=self.classes)
        predicted: list[str] = []
        scores: list[list[float]] = []
        for start in range(0, len(records), MAX_CELLS_PER_CALL):
            chunk = records[start : start + MAX_CELLS_PER_CALL]
            result = self.classify([r["counts"] for r in chunk], names=[r["id"] for r in chunk])
            for p in result["predictions"]:
                predicted.append(p["label"])
                scores.append([p["scores"][c] for c in self.classes])
        return classification_metrics([r["label"] for r in records], predicted, scores, self.classes)

    def adapt(
        self,
        train_records: Sequence[Mapping[str, Any]],
        val_records: Sequence[Mapping[str, Any]] | None = None,
        *,
        classes: Sequence[str] | None = None,
        epochs: int = 4,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        trainable_layers: int = 2,
        weight_decay: float = 0.01,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Bounded gradient fine-tuning of a cell-state classification head on the verified base.

        Builds `BertForSequenceClassification` from the pinned checkpoint (the head is newly
        initialised), freezes every parameter except the head and the last `trainable_layers`
        encoder layers, and runs AdamW for `epochs` passes. Validation records are monitored per
        epoch only; the final epoch's weights are kept (no selection).
        """
        if self.model is None or self.weights_dir is None:
            raise RuntimeError("adapt requires a pipeline built by from_pretrained (no loaded base model)")
        from .samples import validate_dataset

        if not 1 <= int(epochs) <= 50:
            raise ValueError("epochs must be in 1..50 (tutorial-scale adaptation)")
        if not 1 <= int(batch_size) <= MAX_CELLS_PER_CALL:
            raise ValueError(f"batch_size must be in 1..{MAX_CELLS_PER_CALL}")
        if not 0 <= int(trainable_layers) <= 12:
            raise ValueError("trainable_layers must be in 0..12 (the checkpoint has 12 encoder layers)")
        train_manifest = validate_dataset(train_records, vocabulary=self.vocabulary, classes=classes)
        class_list = list(train_manifest["classes"])
        if val_records is not None:
            validate_dataset(val_records, vocabulary=self.vocabulary, classes=class_list)

        import random

        import torch
        from transformers import BertForSequenceClassification

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        clf = BertForSequenceClassification.from_pretrained(
            str(self.weights_dir / MODEL_SUBFOLDER),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            num_labels=len(class_list),
        ).to(self.device)
        for p in clf.parameters():
            p.requires_grad = False
        layers = clf.bert.encoder.layer
        for layer in layers[len(layers) - int(trainable_layers) :] if trainable_layers else []:
            for p in layer.parameters():
                p.requires_grad = True
        for module in (clf.classifier, clf.bert.pooler):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True
        trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in clf.parameters())
        optimizer = torch.optim.AdamW(
            [p for p in clf.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay
        )
        label_index = {c: i for i, c in enumerate(class_list)}
        encoded = [
            (e["tokens"], label_index[r["label"]])
            for r, e in zip(train_records, self.encode([r["counts"] for r in train_records]), strict=True)
        ]
        self.classes = class_list
        self.classifier_model = clf
        self._classifier = self._make_classifier(clf, self.device, self.vocabulary)

        history: list[dict[str, Any]] = []
        for epoch in range(1, int(epochs) + 1):
            clf.train()
            order = list(range(len(encoded)))
            random.shuffle(order)
            total_loss, n_batches = 0.0, 0
            for start in range(0, len(order), int(batch_size)):
                rows = [encoded[i] for i in order[start : start + int(batch_size)]]
                batch = self._batch([tokens for tokens, _ in rows], self.vocabulary, self.device)
                labels = torch.tensor([y for _, y in rows], device=self.device)
                optimizer.zero_grad()
                out = clf(**batch, labels=labels)
                out.loss.backward()
                optimizer.step()
                total_loss += float(out.loss.item())
                n_batches += 1
            clf.eval()
            entry: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(total_loss / max(1, n_batches), 6),
                "n_batches": n_batches,
            }
            if val_records:
                val = self.evaluate(val_records)
                entry["val_accuracy"] = val["accuracy"]
                entry["val_macro_f1"] = val["macro_f1"]
            history.append(entry)
        clf.eval()
        self.adaptation = {
            "method": "gradient fine-tuning (AdamW) of the classification head and pooler"
            + (f" and the last {int(trainable_layers)} encoder layer(s)" if trainable_layers else ""),
            "classes": class_list,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
            "trainable_layers": int(trainable_layers),
            "seed": int(seed),
            "precision": "float32",
            "trainable_parameters": int(n_trainable),
            "total_parameters": int(n_total),
            "trainable_parameter_names": trainable,
            "train_records": len(train_records),
            "val_records": len(val_records) if val_records else 0,
            "selection": "final epoch kept; validation metrics are monitoring only",
            "history": history,
        }
        return dict(self.adaptation)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Export the trainable tensors as safetensors plus a JSON manifest binding them to the base."""
        if self.classifier_model is None or not self.classes:
            raise RuntimeError("save_artifact requires an adapted head (call adapt first)")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adaptation.get("trainable_parameter_names", []))
        tensors = {
            k: v.detach().cpu().contiguous()
            for k, v in self.classifier_model.state_dict().items()
            if k in names
        }
        if not tensors:
            raise RuntimeError("no trainable tensors recorded; nothing to export")
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path))
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "subfolder": MODEL_SUBFOLDER,
                "license": MODEL_LICENSE,
            },
            "classes": list(self.classes),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": digest}
            ],
            "tensors": sorted(tensors),
            "adaptation": {k: v for k, v in self.adaptation.items() if k != "trainable_parameter_names"},
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Rebuild the classification head from an exported artifact (manifest verified before loading)."""
        if self.model is None or self.weights_dir is None:
            raise RuntimeError("load_artifact requires a pipeline built by from_pretrained")
        art = Path(artifact_dir)
        manifest_path = art / ARTIFACT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("model_id"), base.get("model_revision"), base.get("subfolder")) != (
            MODEL_ID,
            MODEL_REVISION,
            MODEL_SUBFOLDER,
        ):
            raise ValueError(
                f"artifact was trained on {base}, this package pins "
                f"{MODEL_ID}@{MODEL_REVISION} ({MODEL_SUBFOLDER})"
            )
        classes = [str(c) for c in manifest.get("classes", [])]
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("artifact manifest must list at least two unique classes")
        for entry in manifest["files"]:
            fp = art / entry["path"]
            if not fp.is_file():
                raise FileNotFoundError(f"artifact file missing: {fp}")
            if fp.stat().st_size != entry["bytes"]:
                raise ValueError(f"{entry['path']}: size {fp.stat().st_size} != manifest {entry['bytes']}")
            if hashlib.sha256(fp.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"{entry['path']}: sha256 mismatch against the artifact manifest")
        from safetensors.torch import load_file
        from transformers import BertForSequenceClassification

        clf = BertForSequenceClassification.from_pretrained(
            str(self.weights_dir / MODEL_SUBFOLDER),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            num_labels=len(classes),
        )
        tensors = load_file(str(art / ARTIFACT_WEIGHTS_NAME))
        if set(tensors) != set(manifest.get("tensors", [])):
            raise ValueError("artifact tensors do not match the names listed in its manifest")
        _missing, unexpected = clf.load_state_dict(tensors, strict=False)
        if unexpected:
            raise ValueError(
                f"artifact carries tensors the base architecture does not have: {sorted(unexpected)[:5]}"
            )
        clf = clf.to(self.device).eval()
        self.classes = classes
        self.classifier_model = clf
        self._classifier = self._make_classifier(clf, self.device, self.vocabulary)
        self.adaptation = {**manifest.get("adaptation", {}), "loaded_from_artifact": str(art)}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> GeneformerPipeline:
        """Verified base snapshot + exported adapter, ready for `classify`."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
