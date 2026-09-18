# Release verification

`tutorials/geneformer_single_cell_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `GeneformerPipeline.from_pretrained(weights_dir=...)`, `validate_dataset`, `split_dataset`, `write_dataset_csv`,
  `pipe.embed`, `majority_baseline`, `library_size_baseline`, `pipe.adapt` with its explicit hyperparameters,
  `pipe.evaluate` on both the validation and the test split, `validate_inputs`, `pipe.classify`,
  `pipe.save_artifact`, `GeneformerPipeline.from_artifact` and the reload-parity assertion), the six expected `outputs/`
  paths, the learner-facing statements (scores are not calibrated probabilities, validation is monitoring only,
  embeddings are representations, the excluded capabilities, homology-aware splitting) and the gated-off BYOD
  default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary
  path, a mutable `revision='main'`, direct `transformers` / `huggingface_hub` / `safetensors` / `RestrictedUnpickler` use **outside the
  carried module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(` without `weights_only=True`,
  `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected backends and temporary manifests, no
weights). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/geneformer-v2-104m/` (the standalone path writes the manifest itself and stages every listed
   file, so the directory may not be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `VAL_FRACTION = 0.2`, `TEST_FRACTION = 0.25`, `SEED = 42`, `EPOCHS = 4`,
   `LEARNING_RATE = 2e-4`, `BATCH_SIZE = 8`, `TRAINABLE_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`,
   `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`;
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `GeneformerPipeline`, `verify_snapshot`, `stage_missing_files`,
     `load_gene_vocabulary`, `RestrictedUnpickler`, `rank_value_encode`, `validate_inputs`,
     `validate_dataset`, `split_dataset`, `generate_sample_dataset`, `write_dataset_csv`,
     `classification_metrics`, `majority_baseline`, `library_size_baseline`) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(..., allow_download=True)`
     reporting the 6 entries fetched from `ctheodoris/Geneformer` at the immutable revision, and
     `verify_snapshot` reporting 6 verified files before the model loads;
   - the three gene dictionaries loaded through the restricted unpickler (20,275 tokens, 42,005 corpus
     medians, 63,675 symbol mappings) and the dataset manifest printed with 64 cells, classes
     `['programme-A', 'programme-B']`, 32/32 class counts, the ceilings and the digest, and the
     splits 36 / 12 / 16;
   - one rank-value encoding read cell by cell, with the ranked heads of the two classes differing;
   - `pipe.embed` reporting 768-dimensional vectors and writing `outputs/geneformer_single_cell_embeddings.csv`;
   - both baselines reported on the test split (majority accuracy 0.5 on the balanced split; the library-size
     baseline near chance, because every cell carries the same library size);
   - `pipe.adapt` reporting the trainable/total parameter counts and a four-epoch history with per-epoch validation
     metrics;
   - `pipe.evaluate` reporting validation and test accuracy, macro-F1, AUROC and per-class rows, and writing
     `outputs/geneformer_single_cell_evaluation_report.json` with both baselines and the delta against the majority baseline;
   - `validate_inputs` accepting the six new sequences and `pipe.classify` writing
     `outputs/geneformer_single_cell_predictions.csv` with per-class scores;
   - `pipe.save_artifact` writing `outputs/geneformer_single_cell_adapter/{adapter.safetensors,manifest.json}`, and
     `GeneformerPipeline.from_artifact` reloading it with identical labels and a maximum absolute score difference below
     `1e-5` (the cell asserts both);
   - `outputs/geneformer_single_cell_result.json` written with `NOTEBOOK_SOURCE`, the model identity, revision and licence,
     the dataset manifest, the evaluation report, the predictions, the artifact manifest and the runtime versions;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache and the weights directory were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `geneformer_single_cell_colab.ipynb` | __LOCAL_ROW__ | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/geneformer_single_cell_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/geneformer_single_cell_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include the model download where it occurred; they are measurements for the stated runtime, not
general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | __LOCAL_ROW__ | Local pre-flight harness (Windows, CPython 3.12, CPU float32) | Default sample path (validate → split → embed → baselines → adapt → evaluate → classify → export → reload) | __LOCAL_WALL__ | **PASSED** — pre-flight; hosted clean-runtime run still required |

## Current status

The notebook source is complete and passes all static checks, including the generator parity checks (`--check` OK).
A local pre-flight execution of the committed blob completed the whole default path on CPU, which catches defects but
is **not** a supported runtime under REL1/REL10. The repository stays at **Candidate** until a Colab or fresh-container
run of the exact release revision is recorded above.
