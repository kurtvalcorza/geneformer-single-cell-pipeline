# Geneformer V2-104M Single-Cell E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/geneformer-single-cell-pipeline`  
**Notebook:** `tutorials/geneformer_single_cell_colab.ipynb`  
**Reviewed commit:** `eca3602e63a650c383ffd364cba200922fd4892b` (`main`, confirmed with `gh api repos/kurtvalcorza/geneformer-single-cell-pipeline/commits/main`)  
**Notebook Git blob:** `792c87c11c03870d239ffa30023442328a2eb021`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-18 (commit `366fc71`); the notebook and the three carried modules have not changed since (the carried modules are byte-identical at the generating revision `53c62e7` and at the reviewed commit).  
**Finding prefix:** `GF`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.

## Executive assessment

This is a technically sound notebook. It implements Geneformer's rank-value encoding in visible Python, stages and digest-verifies a 6-file snapshot (checkpoint plus three gene-dictionary pickles), loads the pickles through an allow-list unpickler, builds a synthetic dataset whose two classes share library size and detected genes so that the library-size baseline is uninformative by construction, fine-tunes a bounded head, evaluates on an untouched test split, and reloads a digest-checked safetensors adapter with exact parity. A CPU run of every code cell in this review reproduced the recorded behaviour, and both documented optional experiments behaved as the notebook says they will.

| Measure | This review (CPU, direct execution) | Kaggle T4 record (blob `792c87c1`) |
|---|---|---|
| Code cells completed | 14/14 at defaults, 235.5 s (install cell with the CI pre-installed switch; torch 2.13.0+cpu, not the 2.14.0 pin) | 14/14 after a restart following the install cell, 279.4 s |
| Snapshot | 6 files fetched and verified (421,180,349 bytes) | 6 files |
| Split / dataset digest | 36 / 12 / 16, digest `f6ac3ca3…` | 36 / 12 / 16 |
| Baselines (test) | majority 0.5 / macro-F1 0.3333; library size 0.5 / AUROC 0.4922 | 0.5 / 0.5 |
| Validation accuracy per epoch | 0.5, 0.5, 1.0, 1.0 | not recorded in the repository |
| Test accuracy / macro-F1 / AUROC (n=16) | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| Reload parity | max abs score diff 0.0 | 0.0 |
| Optional experiment `EPOCHS = 2` | test accuracy 0.5, AUROC 1.0 (the under-trained regime, as promised) | not run |
| Optional experiment `TRAINABLE_LAYERS = 0` | 592,130 trainable; test accuracy 1.0 | not run |
| Smallest balanced two-class BYOD set that passes all stages | **18 cells per class** (stated minimum: 8 records, 3 per class) | not run |

Two problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (GF-M1).** The notebook pip-installs exact pins (including `torch==2.14.0` and `numpy==2.5.3`) into the running kernel and stops with a restart instruction if a loaded distribution changed. The only hosted run needed that restart; `docs/release-verification.md` calls it "expected", and the repository is marked `Release-grade` on that run.
2. **Guided layer largely absent (GF-M2).** Declared `GUIDED`, but there is no intended-learner statement, how-to-use, roadmap, glossary, prediction prompt, checkpoint with sample answer, troubleshooting section or conclusion template, and 1,287 lines of carried modules sit in three cells that are not labelled as infrastructure the learner may skip.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites (cell 1, "Knowledge"): what a single-cell count matrix is, why library size differs between cells, how accuracy, macro-F1 and AUROC differ |
| Supported runtime | "a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU is enough … CUDA is used automatically when present" |
| Promised outcomes | Pinned install; carried package; staged, digest-verified snapshot; dictionaries via restricted unpickler; deterministic 64-cell dataset validated and split 36/12/16; one rank-value encoding read; 768-d embeddings; majority and library-size baselines; bounded AdamW fine-tuning of head + pooler + last two layers; held-out accuracy / macro-F1 / AUROC; six new cells classified; safetensors adapter export and reload parity; BYOD (CSV / JSON / JSONL) through the same stages |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `53c62e7`; `--check` passes at the reviewed commit |
| Release status | **`Release-grade`** (`STATUS.md`, `README.md`, `tutorials/README.md`, `docs/release-verification.md` Current status) |

### Evidence actually obtained

- **Source inspection.** All 31 cells (14 code; cells 5, 7, 9 are the carried `metrics.py` 158 lines, `pipeline.py` 774 lines, `samples.py` 355 lines). Also read: `pipeline.py` (`adapt`, `evaluate`, `rank_value_encode`, `RestrictedUnpickler`), `samples.py` (`validate_dataset`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv`), `metrics.py` (`library_size_baseline`); `tools/build_notebook.py` (install guard, `module_sha256`) and the relevant lines of `tools/notebook_template.py`; `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/` directory.
- **Documented execution evidence.** `docs/release-verification.md`, Kaggle batch kernel `dimer-nb2-geneformer-single-cell` v1, 2026-09-18, **the reviewed blob**, Python 3.12.13, Tesla T4, clean Hugging Face cache, no checkout: 14/14 cells "after the expected fresh-process restart following dependency installation", 279.4 s. The executed notebook of that run is not archived in the repository or the workspace backups I searched, so the restart is taken from the record's own text. No Colab run of this blob; no CPU hosted run although CPU is the documented default runtime; no BYOD or optional-experiment run.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only, Python 3.12.14, torch 2.13.0+cpu, transformers 4.57.6, safetensors 0.8.0, huggingface-hub 0.36.2, numpy 2.5.3 (shared read-only conda env; nothing installed). The install cell ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, so it skipped pip; the rest of the cells ran unmodified in one namespace from an empty working directory with an empty HF cache, under a `google.colab` shim.
  - **Probes (about 8 min total):** P1 static structure and guided-layer markers; P2 generator `--check` and module provenance; P3 all 14 code cells at defaults (real 421 MB download); P4 the two documented optional experiments, each re-run from Section 8 (cell 21) to the end; P5 the BYOD branch of cell 13 with a shimmed upload (template CSV round trip, five invalid files, a cancelled upload) and the split contract for small datasets.
- **Not verified:** the pinned install itself and therefore the restart (GF-M1) beyond the Kaggle record; any Colab run; GPU behaviour; the real upload dialog; BYOD beyond cell 13 with a real dataset.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** strong. Supply chain (pinned revision, per-file SHA-256, `local_files_only`, `trust_remote_code=False`, `use_safetensors=True`), the pickle allow-list, the adapter manifest checked before deserialisation, and `adapt` rebuilding from the verified base on every call (so a rerun never continues from adapted weights; confirmed by P4). Defects: the install pattern forces a restart (GF-M1); the per-split validation in `adapt`/`evaluate` makes the stated BYOD minimum fail late (GF-m1); two provenance labels are imprecise (GF-m2).
- **Scientific validity:** good for a tutorial. Library size and detected genes are equalised by construction, so the library-size baseline lands at chance (0.5, AUROC 0.49) and the model's 1.0 is attributable to rank order; validation is monitoring only and the final epoch is kept; the Interpretation section says plainly that the classes are a generator rule, not biology, and that real data needs donor/batch-level splits. Saturated 1.0 on 16 synthetic cells is expected and labelled as sample-sanity evidence.
- **Promise fulfilment:** every promised default stage executes and produces its output (P3). The BYOD contract is stated more loosely than the code enforces (GF-m1).
- **Learner experience:** the conceptual explanation of rank-value encoding is clear and the "Look for" notes match what runs (14.8 M trainable parameters; validation accuracy leaving 0.5 at epoch 3). The guided scaffolding is missing (GF-M2), the optional experiments give no rerun scope or prediction step (GF-m3), and the embedding inspection prints two near-identical numbers with no reading guidance (GF-m4).
- **Spec conformance:** RUN1, RUN10, ENV6, REL2, REL11 unmet (GF-M1, all `MUST`); DAT12 and DAT19 unmet for the BYOD minimum (GF-m1, `MUST`); SRC3 (stale instruction in the release procedure and a template artefact, GF-m2); GDL1–GDL14 largely unmet (`SHOULD`, GF-M2); declares 2.0 (GF-S1).

## 3. Promise and objective tracing

| Claim | Implementation | Observable result (P3) | Learner interpretation |
|---|---|---|---|
| Run all installs the pinned runtime and completes | cell 3 pip install + stale-module guard | Kaggle: restart required; not run here | Not delivered as one pass (GF-M1) |
| Snapshot staged and digest-verified before loading | cell 11 | 6 fetched, 6 verified, then load | Clear |
| Dictionaries loaded as data | `RestrictedUnpickler` in cell 7, used by `from_pretrained` | 20,275 tokens / 42,005 medians / 63,675 symbols | Clear; explained in Section 4 |
| Library size carries no signal | `generate_sample_dataset` | library 19,987–20,010; 348 genes in every cell | Clear |
| Rank-value encoding read cell by cell | cell 15 | different top-8 heads, same genes detected | Clear |
| Embeddings as representations | cell 17 | 768-d; within 0.9961 vs between 0.9948 | Numbers printed without a reading guide (GF-m4) |
| Baselines fitted on train only | cell 19 | 0.5 / 0.5 (AUROC 0.4922) | Clear |
| Bounded fine-tuning | cell 21 | 14,767,874 of 104,366,594; val 0.5→1.0 at epoch 3 | Matches the "Look for" note |
| Held-out evaluation | cell 23 | test 1.0 / 1.0 / 1.0, n=16; report written | Labelled tutorial evidence |
| New-cell inference with uncalibrated scores | cell 25 | 6/6, scores ≈0.998 | Clear |
| Adapter export and fresh reload | cell 27 | 36 tensors, 59,075,552 bytes; parity 0.0 | Clear |
| BYOD through the same stages, stated minimum 8 records / 3 per class | cells 13 → 27 | a 4+4 set passes cell 13 and fails in `adapt` | Contract misstated (GF-m1) |

| Objective (cell 0) | Learner activity | Evidence it was exercised |
|---|---|---|
| install the pinned runtime; inspect the carried modules | run cells | run only; no inspection task |
| stage/verify snapshot; restricted unpickler | read printed counts | observation only |
| validate and split without leakage | read the manifest | observation only |
| read a rank-value encoding | compare two printed heads | closest to an activity; no question asked |
| extract embeddings; measure baselines; fine-tune; evaluate; classify; export | run cells, read numbers | observation only; optional experiments exist but have no predict/explain step (GF-m3) |

## 4. Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Strong conceptual introduction and accurate "Look for" notes; guided layer absent (GF-M2); carried code not marked skippable; embedding numbers unexplained (GF-m4); `{{id, counts, label}}` artefact in Prerequisites (GF-m2) |
| Clean default | Documented (Kaggle T4, reviewed blob) + direct CPU execution | Kaggle: 14/14 only after a restart (GF-M1). CPU here: 14/14, 235.5 s, all seven outputs written, numbers as recorded. Install cell not exercised with pip |
| Active learning | Direct CPU execution | Both documented experiments run from Section 8 and behave as promised (EPOCHS=2: accuracy 0.5 / AUROC 1.0; TRAINABLE_LAYERS=0: 592,130 trainable, accuracy 1.0). The notebook does not say which cells to re-run or ask for a prediction (GF-m3) |
| Reuse and recovery | Direct CPU execution (cell 13 only) | Template CSV round-trips; mouse ids, bad header, one class and non-numeric values give actionable messages; stated minimum fails late in `adapt` (needs 18 per class); `.h5ad` gives `UnicodeDecodeError`; cancelled upload gives bare `StopIteration`; BYOD records overwrite `…sample_dataset.csv`; BYOD needs `google.colab` although Jupyter is a supported runtime (GF-m1). Downstream BYOD stages with real data not verified |

## 5. Findings

### Major

#### GF-M1 — `Run all` needs a manual restart after the install cell, and the blob is marked `Release-grade` on that run

- **Cell/section:** cell 3 (Section 1); `tools/build_notebook.py` `_INSTALL_GUARD` (lines 44–60) and the install-cell emitter; `docs/release-verification.md` (manual evidence row and recorded executions row of 2026-09-18).
- **Observed issue:** the cell records every distribution imported in the kernel, runs `pip install -q` of seven exact pins into that kernel (`torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`), and raises `RuntimeError(... Restart the runtime, then rerun from the top.)` if any loaded distribution changed. On a hosted image whose kernel has already imported NumPy or torch at other versions this always fires on the first pass.
- **Consequence:** a learner choosing **Run all** stops at cell 3 and has to restart and run again; that is exactly the "second manual execution after package installation" §5 says is not `Run all` conformant. The repository still reports a `Run all` PASS and `Release-grade` for it.
- **Evidence:** documented — `docs/release-verification.md`: "14/14 code cells after the expected fresh-process restart following dependency installation" and "One expected fresh-process restart followed the install cell" (Kaggle T4, blob `792c87c1`). Source — cell 3. Direct — not exercised (pip skipped by the CI switch).
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementation on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/build_notebook.py` / `tools/notebook_template.py`, regenerate, re-qualify with a one-pass hosted Run all, and correct `docs/release-verification.md` (and the status token in `STATUS.md`, `README.md`, `tutorials/README.md`) so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Colab (or Kaggle) runtime executes the regenerated notebook's code cells in order in one kernel with no restart and no error output; the recorded run says so explicitly; `docs/release-verification.md` no longer describes a restart as expected.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11 (all `MUST`).

#### GF-M2 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** cells 0–2, 4–9, 30; `tools/notebook_template.py` opening and closing blocks.
- **Observed issue:** no intended-learner statement (GDL1), no **How to use this notebook** (GDL2), no roadmap (GDL3), no glossary although macro-F1, AUROC, softmax calibration, pooler, AdamW and corpus median appear (GDL6), no question or prediction before the baseline/fine-tuning comparison (GDL7), no interpretation checkpoint with a sample answer (GDL9), no Predict → Change → Run → Observe → Explain activity (GDL10), no troubleshooting section for download, memory, pickle-refusal or BYOD errors (GDL13), and no conclusion template (GDL14). Learning objectives are mostly "install / inspect / run" rather than observable verbs (GDL5). The three carried-module cells (1,287 lines) are introduced as "the repository's package" and are neither labelled infrastructure the learner may run without studying nor collapsed (GDL11/GDL12). P1 found none of these markers except "Look for" notes.
- **Consequence:** a self-paced learner gets an excellent technical narrative but is never asked to predict, interpret or conclude anything; the objectives are exercised only by reading output.
- **Evidence:** source inspection; P1 `guided_markers`.
- **Recommended correction:** add the GDL layer in the template: audience and how-to-use block, a five-line roadmap, an Input → Model → Output contract (cell counts → rank-value tokens → encoder → embedding / class scores), a short glossary, a prediction prompt before Section 7/8 ("what accuracy will the library-size baseline reach, and why?") with a collapsible answer, a checkpoint after Section 9, a troubleshooting table, and a conclusion template in the Interpretation section; mark cells 5/7/9 **Infrastructure — you may run these without reading them** and set `collapsed`/`jupyter.source_hidden` metadata.
- **Acceptance check:** the regenerated notebook contains each GDL1–GDL14 element listed above (identifiable by heading or callout), the carried cells carry an infrastructure label and collapse metadata, and `--check` and the parity tests pass.
- **Spec:** GDL1–GDL7, GDL9–GDL14 (`SHOULD`); UX8.

### Minor

#### GF-m1 — BYOD: the stated minimum fails late, and several failures are not actionable

- **Cell/section:** cells 1 and 13 (Prerequisites "Data contract", Section 4); `samples.py` `validate_dataset`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv`; `pipeline.py` `adapt`/`evaluate`.
- **Observed issue:** (a) the Prerequisites state "at least 8 records and 3 per class". Cell 13 accepts such a set, but `adapt` and `evaluate` re-validate **each split** against the same 8-record / 3-per-class minimum, so a 4+4 set fails in Section 8 with `dataset has 4 records; at least 8 are required`, after the dataset was accepted and without naming the split. The smallest balanced two-class set that passes every stage at the default fractions is **18 cells per class** (P5). (b) Prerequisites name Jupyter as a supported runtime, but the BYOD branch imports `google.colab` and has no location field. (c) An `.h5ad` upload fails with `UnicodeDecodeError` instead of the loader's own "unsupported BYOD file type" message (the file is decoded before the suffix is checked). (d) A cancelled upload raises a bare `StopIteration`. (e) Cell 13 writes the user's records to `outputs/geneformer_single_cell_sample_dataset.csv`, the file the notebook calls the sample template.
- **Consequence:** a learner who sizes a dataset by the stated contract loses a model load and a fine-tuning start before learning the real minimum; Jupyter users cannot use BYOD at all.
- **Evidence:** direct execution P5 (`stated_minimum_8x2cls`, `smallest_two_class_balanced_passing_all_splits`, `invalid_cells.h5ad`, `upload_cancelled`, `template_overwritten_by_byod_records`); mouse ids, bad header, one class and non-numeric values all gave actionable messages. Source for (b).
- **Recommended correction:** validate the split sizes in cell 13 right after `split_dataset` (or make the per-split minimum explicit and smaller) and state the real minimum in Prerequisites and Section 4; add a `BYOD_PATH` form field read without `google.colab` (EXE2); check the suffix before decoding; replace the bare `next(iter(...))` with a named "no file uploaded" error; write BYOD records to a separate file name.
- **Acceptance check:** a dataset at the stated minimum either completes Sections 4–11 or is refused in cell 13 with a message naming the split and the required size; `BYOD_PATH` set to a local CSV works with no `google.colab` import; `.h5ad` and a cancelled upload each produce a message naming the condition and the fix; the sample template file is not overwritten by BYOD.
- **Spec:** DAT12, DAT19 (`MUST`); EXE2, UX10 (`SHOULD`).

#### GF-m2 — Stale instruction and imprecise provenance labels

- **Cell/section:** cell 1; cell 3 `NOTEBOOK_SOURCE`; `docs/release-verification.md` procedure step 3; `tools/notebook_template.py` line 110; `tools/build_notebook.py` line 301.
- **Observed issue:** (a) Prerequisites render `{{id, counts, label}}` with doubled braces (a format-string escape in a template line that is not formatted). (b) The release procedure says the form defaults include `BATCH_SIZE = 8`; the notebook default is `4` (cell 21, and the run recorded `batch_size: 4`). (c) `NOTEBOOK_SOURCE` and `metadata.dimer.generated_from` pair `'embedded_module': 'src/…/pipeline.py'` with `'module_sha256': 'e27aa254…'`, but that digest is the hash of the three modules concatenated; `pipeline.py` itself hashes to `9ca2847a…` (the per-cell tags are correct).
- **Consequence:** small, but a verifier following the procedure would set the wrong batch size, and a reader checking the exported digest against `pipeline.py` would see a mismatch.
- **Evidence:** source inspection; P1 `double_brace_in_markdown: [1]`; P2 digests.
- **Recommended correction:** use single braces in the template line; correct the procedure to `BATCH_SIZE = 4`; rename the exported field (for example `modules_sha256_concat`) or export `per_module_sha256`.
- **Acceptance check:** cell 1 renders `{id, counts, label}`; the procedure's defaults equal the notebook's form fields; every exported digest labelled with a file name equals that file's SHA-256.
- **Spec:** SRC3 (`MUST`); OUT7 (`SHOULD`).

#### GF-m3 — The optional experiments give no rerun scope and no prediction step

- **Cell/section:** cell 30, "Optional experiments".
- **Observed issue:** the experiments say what to change (`TRAINABLE_LAYERS = 0`, `EPOCHS = 2`, BYOD) but not which cells to re-run, and do not ask the learner to predict the outcome first. Re-running only cell 21 leaves the evaluation, predictions, adapter and result files from the previous configuration.
- **Consequence:** a learner can compare stale Section 9–12 outputs with a new training history; the one activity that could exercise the objectives stays a "change and look" step.
- **Evidence:** direct execution P4 — re-running cells 21–29 gives a clean, correct result for both experiments (EPOCHS=2: test accuracy 0.5, AUROC 1.0; TRAINABLE_LAYERS=0: 592,130 trainable parameters, accuracy 1.0), because `adapt` rebuilds from the base each time. Source for the missing scope.
- **Recommended correction:** state "re-run from Section 8 to the end"; frame one experiment as Predict → Change → Run → Observe → Explain with a collapsible expected answer (the EPOCHS=2 case is ideal: accuracy at chance with AUROC 1.0).
- **Acceptance check:** the optional-experiments text names the cells to re-run and includes a prediction prompt and an explanation answer for at least one experiment.
- **Spec:** GDL10, UX5 (`SHOULD`).

#### GF-m4 — The embedding inspection prints two near-identical numbers with no reading guide

- **Cell/section:** cells 16–17 (Section 6).
- **Observed issue:** the cell prints mean cosine within class 0.9961 and between classes 0.9948 (P3). The markdown says only that this is "an inspection, not an evaluation"; it does not say what to expect or how to read a 0.0013 difference, and offers no plot.
- **Consequence:** a learner may conclude either that the embeddings separate the classes or that the model sees no difference; neither is supported, and the point (frozen mean-pooled embeddings are dominated by shared genes, which is why a trained head is needed) is left unstated.
- **Evidence:** direct execution P3; source.
- **Recommended correction:** add a "What to notice" note explaining that all cells are highly similar under mean pooling and that the small gap is not an evaluation; optionally show a 2-D projection of the eight vectors.
- **Acceptance check:** Section 6 states what the two numbers mean and why they are close, without hard-coding the values.
- **Spec:** GDL8, UX4, UX11 (`SHOULD`).

### Suggestions

- **GF-S1** — Regenerate against NOTEBOOK_SPEC 2.2 (the notebook, metadata and registry declare 2.0).
- **GF-S2** — Document `DIMER_NOTEBOOK_CI_PREINSTALLED` in the notebook (EXE5); it is read by cell 3 and unexplained.
- **GF-S3** — Record a hosted **CPU** run: CPU is the documented default runtime ("CPU is enough"), but the only hosted record is on a Tesla T4.
- **GF-S4** — Show the per-epoch training history as a small table or plot so the epoch-3 jump in validation accuracy is visible next to the under-trained-regime explanation.

## 6. Readiness

**Needs revision.** Open Majors: GF-M1 (one-pass `Run all`; RUN1/RUN10/ENV6/REL2/REL11 `MUST`) and GF-M2 (guided layer). Unresolved applicable `MUST`s also include DAT12/DAT19 (GF-m1) and SRC3 (GF-m2). Remaining gates after fixes: a one-pass hosted Run all of the regenerated blob recorded in `docs/release-verification.md`, and a BYOD check per REL12.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, this review):** all 14 code cells at defaults with the install skipped; the recorded numbers (split, baselines, 14,767,874 trainable parameters, test 1.0, reload parity 0.0); both optional experiments' outcomes; the BYOD minimum arithmetic and the five invalid-input messages in cell 13.
- **From documented evidence only:** the restart after the install cell on Kaggle T4 (the executed notebook of that run was not available to me).
- **Inferred:** that a Colab runtime also triggers the restart (it preloads NumPy at a different version from the `numpy==2.5.3` pin); that real single-cell BYOD data completes Sections 8–11.
- **Only Kurt can confirm:** whether GF-M2's guided-layer gap is to be fixed now or with the fleet-wide 2.2 migration.
- **Most likely to be wrong:** GF-M1's mechanism on Colab. It rests on the Kaggle record's wording and on Colab preloading a different NumPy; if a current Colab image happened to match every pin already loaded, the guard would not fire there, although the Kaggle record would still stand.
