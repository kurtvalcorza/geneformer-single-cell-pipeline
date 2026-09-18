# Weight provenance and DIMER hosting

This repository pins **one** snapshot with its own `dimer-base-manifest.json`. That snapshot carries both the checkpoint and the gene dictionaries Geneformer needs to tokenize a cell, because they live in the same upstream repository at the same revision.

## Geneformer V2-104M weights and dictionaries

- Upstream: `ctheodoris/Geneformer`, subfolder `Geneformer-V2-104M`
- Immutable revision: `1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5`
- Weight format: SafeTensors (`Geneformer-V2-104M/model.safetensors`, 417,571,156 bytes)
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream repository metadata)
- Local layout: `weights/geneformer-v2-104m/` holds the 6 manifest entries (`Geneformer-V2-104M/config.json`, `Geneformer-V2-104M/model.safetensors`, `geneformer/token_dictionary_gc104M.pkl`, `geneformer/gene_median_dictionary_gc104M.pkl`, `geneformer/gene_name_id_dict_gc104M.pkl`, upstream `README.md`; 421,180,349 bytes total) with byte size and SHA-256 for each. `verify_snapshot()` in `src/geneformer_single_cell_pipeline/pipeline.py` checks all of them before any load; `stage_missing_files(allow_download=True)` fetches only absent entries at the pinned revision. `.safetensors` and `.pkl` files are git-ignored: the Git repository vendors neither the checkpoint nor the three dictionary pickles, so a clone carries the manifest and reproduces the bytes from the pinned revision.
- Cross-check: the manifest digests for the checkpoint and the three dictionaries equal the `oid sha256` values of their Hub LFS pointers at the pinned revision.

## Dictionary trust boundary

Geneformer's tokenizer is not a vocabulary file — it is three Python **pickles**: gene id → token, gene id → corpus median expression, gene symbol → gene id. A pickle is code-capable by construction, so this pipeline does two things rather than one:

1. the dictionaries are part of the digest-verified snapshot, so their bytes are pinned; and
2. they are loaded through `RestrictedUnpickler`, whose `find_class` refuses every global except `numpy.core.multiarray.scalar` / `numpy._core.multiarray.scalar`, `numpy.dtype` and the concrete `numpy.dtypes.*DType` classes that the median dictionary legitimately contains. Anything else raises `UnpicklingError` naming the refused symbol — a test pickles a callable and asserts the refusal, and the recorded smoke run refused `nt.getcwd`.

The loaded dictionaries are then shape-checked: the token dictionary must hold exactly 20,275 entries (matching `config.json`'s `vocab_size`) and all four special tokens, or loading fails. Path-safe loading of a pinned pickle is not a claim that pickles are safe in general; it is a narrow, verified refusal surface for these exact bytes.

## Files deliberately not staged

- `Geneformer-V2-104M/training_args.bin` — a torch pickle of training arguments; not needed for inference or adaptation.
- `Geneformer-V2-104M/generation_config.json` — this pipeline does not generate.
- Every other checkpoint in the repository: `Geneformer-V1-10M`, `Geneformer-V2-316M`, `Geneformer-V2-104M_CLcancer`, and the 316M model in the repository root. The subfolder is part of this package's model identity and of every exported artifact's manifest.

A DIMER profile upload for this model should use `Geneformer-V2-104M/model.safetensors` together with the three dictionaries, because the checkpoint cannot tokenize a cell without them.

## DIMER hosting

- Apache-2.0 permits use, modification, redistribution and commercial use subject to preservation of the licence and notices. DIMER may mirror the pinned snapshot in its model store under those terms; the weights are redistributed unmodified.
- Loader trust boundary: `transformers` 4.57.6 builds the native `BertModel` / `BertForSequenceClassification` classes for this checkpoint. The pipeline passes `trust_remote_code=False` and `local_files_only=True`, so no upstream Python is executed and no file is resolved from the Hub or an HF cache on the snapshot path. The upstream `geneformer` package is not installed or imported.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
