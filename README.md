# Geneformer Single-Cell Pipeline

DIMER-oriented pipeline for **Geneformer V2-104M** (`ctheodoris/Geneformer`, subfolder `Geneformer-V2-104M`), pinned to an immutable Hugging Face revision. The repository implements Geneformer's rank-value encoding in ordinary Python, and exposes cell embeddings, a labelled-cell dataset contract with explicit ceilings, bounded gradient fine-tuning of a cell-state classification head, held-out metrics with trivial baselines, and a safetensors adapter artifact that is digest-verified before it is loaded.

## Upstream alignment

- Model: `ctheodoris/Geneformer`, subfolder `Geneformer-V2-104M`
- Revision: `1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5`
- Upstream weight license: Apache-2.0
- Upstream task: single-cell transcriptome foundation model (masked gene modelling); this repository uses the encoder for cell embeddings and cell-state classification
- Repository adaptation: **E2E** (bounded fine-tuning of the classification head plus the last *n* encoder layers, with a portable safetensors adapter)

## Quick start

```python
from geneformer_single_cell_pipeline import GeneformerPipeline, generate_sample_dataset, split_dataset

pipe = GeneformerPipeline.from_pretrained()        # verifies the snapshot and loads the gene dictionaries
cells = generate_sample_dataset(pipe.vocabulary)   # synthetic cells over real human Ensembl ids
print(pipe.embed([cells[0]["counts"]])["dimension"])   # 768

splits = split_dataset(cells)
pipe.adapt(splits["train"], splits["validation"])      # bounded AdamW fine-tuning
print(pipe.evaluate(splits["test"])["accuracy"])
print(pipe.classify([cells[0]["counts"]])["predictions"][0]["label"])
```

A cell is a mapping from a human Ensembl gene id (`ENSG...`) or an approved gene symbol to a non-negative count. `rank_value_encode` normalises the counts, divides each gene by its corpus median, ranks the genes and maps them to tokens — and reports every gene it dropped (zero count, no token, no median) and every low-rank gene it truncated at the 4,094-token input limit. At least 10 genes per cell must be detected *and* encodable, at most 32 cells per call. Datasets are `{id, counts, label}` records: at least 8 rows and 3 per class, at most 2,000 rows and 20 classes, unique ids.

## Weights layout

```
weights/geneformer-v2-104m/Geneformer-V2-104M/   config.json  model.safetensors
weights/geneformer-v2-104m/geneformer/           token_dictionary_gc104M.pkl
                                                 gene_median_dictionary_gc104M.pkl
                                                 gene_name_id_dict_gc104M.pkl
weights/geneformer-v2-104m/                      README.md  dimer-base-manifest.json
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` (byte size + SHA-256 of every manifest entry; staging fetches only absent entries, only at the pinned revision, and only with `allow_download=True`), loads the three dictionaries through `RestrictedUnpickler`, and then loads the model with `local_files_only=True`, `trust_remote_code=False` and `use_safetensors=True`. The upstream `training_args.bin` and the repository's other checkpoints are deliberately absent from the manifest and are never loaded. `.safetensors` and `.pkl` files are git-ignored: the repository vendors neither the checkpoint nor the dictionary pickles, and a clone reproduces both from the pinned revision. See `docs/WEIGHTS.md`.

## The dictionaries are pickles — and are loaded as data

Geneformer's tokenizer ships as three Python pickles. They are pinned and digest-verified like any other snapshot file, and then read through `RestrictedUnpickler`, which refuses every global except the numpy scalar and dtype classes the median dictionary legitimately contains. An unexpected class raises `UnpicklingError` naming the refused symbol instead of executing it, and the loaded token dictionary is shape-checked against the `vocab_size` in `config.json`.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the trained tensors only — head, pooler and the unfrozen encoder layers) plus a `manifest.json` recording the artifact format, the exact base model id, revision **and subfolder**, the class order, the tensor names, the file size and SHA-256, and the full adaptation configuration. `GeneformerPipeline.from_artifact(dir)` re-verifies the base snapshot, then checks the artifact manifest, the base identity and every digest **before** deserialising, and refuses any tensor the base architecture does not have.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: injected backends, a stand-in vocabulary and temporary manifests, never the weights.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/geneformer-single-cell-pipeline/blob/main/tutorials/geneformer_single_cell_colab.ipynb)

`tutorials/geneformer_single_cell_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 3 package modules verbatim in dependency order, the pinned model identity, the snapshot manifest and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable. Do not edit it by hand; change the package or the template and regenerate (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI). Its default path stages and digest-verifies the snapshot, loads the gene dictionaries through the restricted unpickler, generates and validates a deterministic 64-cell dataset whose two classes share their library size and detected genes, splits it 36/12/16 stratified by class, reads one rank-value encoding cell by cell, extracts cell embeddings, measures majority-class and library-size baselines, runs a 4-epoch bounded fine-tuning, evaluates accuracy/macro-F1/AUROC on the held-out test split, classifies six freshly generated cells, exports the safetensors adapter and verifies reload parity. BYOD (CSV/JSON/JSONL) is optional and gated off by default. See `tutorials/README.md`.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. One local CPU pre-flight execution of the committed notebook is recorded in `docs/release-verification.md`; complete the supported clean-runtime procedure in that document against the exact release revision before calling the notebook release-grade.

## Licensing

- Upstream weights and dictionaries: Apache-2.0 (`ctheodoris/Geneformer`), redistributed unmodified from the pinned revision.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
