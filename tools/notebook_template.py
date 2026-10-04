"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E single-cell workflow: the pinned Geneformer V2-104M snapshot is
digest-verified (weights *and* the three gene dictionaries), cells are rank-value encoded in the
notebook, a synthetic order-sensitive dataset is validated and split, trivial baselines are
measured, a bounded AdamW fine-tuning runs, and the adapter is exported and reloaded.

Review fixes (Notebook Review Framework v1, review PR #7, GF-M1..M2 / GF-m1..m4): the runtime is the fleet's uv
isolated environment (no in-kernel install, no restart); the guided layer (who it is for, input/model/output, how to
use, roadmap, predictions, worked answers, a change-one-thing activity, troubleshooting, glossary, conclusion) is
added and the setup cells are labelled Infrastructure and collapsed; BYOD splits are validated in Section 4 with the
minimums `adapt` and `evaluate` apply, a dataset change drops any earlier head, `BYOD_PATH` reads a file without
`google.colab`, an unsupported or cancelled upload gets an actionable message, and BYOD data is written under its
own name; the optional experiments name the cell to re-run from; and Section 6 explains the two cosine numbers.

Code cells are written with single braces and escaped by ``_py`` for the generator's ``str.format`` pass; ``@STEM@``
becomes the output stem. Markdown cells are formatted too, so they contain no literal braces.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "geneformer-single-cell-pipeline"


def _py(code: str) -> str:
    """Escape a code cell for the generator's ``str.format`` pass; ``@STEM@`` stands for ``{stem}``."""
    return code.replace("{", "{{").replace("}", "}}").replace("@STEM@", "{stem}")


BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/geneformer_single_cell_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ctheodoris%2FGeneformer-ffcc4d?style=flat",
        "https://huggingface.co/ctheodoris/Geneformer",
    ),
    (
        "Nature 2023",
        "https://img.shields.io/badge/Nature-Transfer%20learning%20in%20network%20biology-b31b1b.svg",
        "https://www.nature.com/articles/s41586-023-06139-9",
    ),
    (
        "bioRxiv 2024",
        "https://img.shields.io/badge/bioRxiv-2024.08.16.608180-b31b1b.svg",
        "https://doi.org/10.1101/2024.08.16.608180",
    ),
]

_DATA_CODE = _py(
    """import json
import os
from pathlib import Path

USE_BYOD = False  # @param {type:"boolean"}
BYOD_PATH = ''  # @param {type:"string"}
VAL_FRACTION = 0.2  # @param {type:"number"}
TEST_FRACTION = 0.25  # @param {type:"number"}
SEED = 42  # @param {type:"integer"}

os.makedirs('outputs', exist_ok=True)
# A new dataset needs a new head: drop any head an earlier run trained, so no later cell can evaluate,
# classify or export it under this dataset's name.
pipe.reset_adaptation()
vocabulary = pipe.vocabulary
print({'tokens': len(vocabulary.tokens), 'corpus_medians': len(vocabulary.medians), 'gene_symbols': len(vocabulary.symbol_to_id), 'encodable_gene_ids': len(vocabulary.gene_ids), 'special_tokens': vocabulary.special})

if USE_BYOD:
    if BYOD_PATH:
        byod_path = Path(BYOD_PATH)
        file_name = byod_path.name
    else:
        try:
            from google.colab import files
        except ImportError:
            raise RuntimeError('The upload dialog exists only on Google Colab. Set BYOD_PATH to your .csv, .json or .jsonl file and run this cell again.') from None
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError(f'Upload exactly one .csv, .json or .jsonl file (got {len(uploaded)}; an empty or cancelled upload gives 0). Run this cell again, or set BYOD_PATH to the file instead.')
        file_name, payload = next(iter(uploaded.items()))
        byod_path = Path('work') / file_name
        byod_path.parent.mkdir(parents=True, exist_ok=True)
        byod_path.write_bytes(payload)
    records = load_byod_dataset(byod_path)
    data_source = 'BYOD (' + file_name + ')'
    dataset_csv = 'outputs/@STEM@_byod_dataset.csv'
else:
    records = generate_sample_dataset(vocabulary)
    data_source = f'synthetic median-matched programme dataset (seed {SAMPLE_SEED}, {SAMPLE_SIZE} cells)'
    dataset_csv = 'outputs/@STEM@_sample_dataset.csv'

dataset_manifest = validate_dataset(records, vocabulary=vocabulary)
CLASSES = dataset_manifest['classes']
splits = split_dataset(records, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED)
# Every split must hold every class at least once: the minimum adapt and evaluate apply. A refusal names the split.
split_manifests = validate_splits(splits, CLASSES, vocabulary=vocabulary)
train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']
write_dataset_csv(records, dataset_csv)

print({'data_source': data_source, 'n_records': dataset_manifest['n_records'], 'classes': CLASSES, 'class_counts': dataset_manifest['class_counts']})
print({'detected_genes': dataset_manifest['detected_genes'], 'encodable_genes': dataset_manifest.get('encodable_genes'), 'library_size': dataset_manifest['library_size']})
print({'ceilings': dataset_manifest['ceilings'], 'digest': dataset_manifest['digest'][:16] + '...'})
print({name: {'n': m['n_records'], 'class_counts': m['class_counts']} for name, m in split_manifests.items()})
print({'written': dataset_csv})"""
)

_ENCODE_CODE = _py(
    """example_a = next(r for r in records if r['label'] == CLASSES[0])
example_b = next(r for r in records if r['label'] == CLASSES[1])
encoded_a = rank_value_encode(example_a['counts'], vocabulary)
encoded_b = rank_value_encode(example_b['counts'], vocabulary)

def summarise(record, encoded):
    return {
        'id': record['id'],
        'label': record['label'],
        'library_size': encoded['library_size'],
        'detected': encoded['n_detected'],
        'encoded': encoded['n_encoded'],
        'tokens_kept': encoded['n_kept'],
        'truncated': encoded['n_truncated'],
        'unknown_genes': len(encoded['unknown_genes']),
        'genes_without_median': len(encoded['genes_without_median']),
    }

print(summarise(example_a, encoded_a))
print(summarise(example_b, encoded_b))
print('top 8 ranked genes,', example_a['label'] + ':', encoded_a['ranked_gene_ids'][:8])
print('top 8 ranked genes,', example_b['label'] + ':', encoded_b['ranked_gene_ids'][:8])
print('same genes detected in both cells:', set(example_a['counts']) == set(example_b['counts']))
print('first 8 token ids:', encoded_a['tokens'][:8])"""
)

_EMBED_CODE = _py(
    """import csv
import math

embed_records = val_records[:8]
embedding_result = pipe.embed([r['counts'] for r in embed_records], names=[r['id'] for r in embed_records])
vectors = embedding_result['embeddings']
print({'n_cells': embedding_result['n_cells'], 'dimension': embedding_result['dimension'], 'tokens_kept': embedding_result['tokens_kept'][:4], 'pooling': embedding_result['pooling']})

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

within, between = [], []
for i in range(len(embed_records)):
    for j in range(i + 1, len(embed_records)):
        sim = cosine(vectors[i], vectors[j])
        (within if embed_records[i]['label'] == embed_records[j]['label'] else between).append(sim)
print({'mean_cosine_within_class': round(sum(within) / len(within), 4) if within else None, 'mean_cosine_between_classes': round(sum(between) / len(between), 4) if between else None, 'note': 'inspection only; embeddings are unlabelled representations'})

with open('outputs/@STEM@_embeddings.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'label'] + [f'dim_{k}' for k in range(embedding_result['dimension'])])
    for r, vec in zip(embed_records, vectors):
        writer.writerow([r['id'], r['label']] + [f'{x:.6f}' for x in vec])
print('wrote outputs/@STEM@_embeddings.csv')"""
)

_BASELINE_CODE = _py(
    """baseline_majority = majority_baseline(train_records, test_records, CLASSES)
print({k: baseline_majority[k] for k in ('baseline', 'predicted_label', 'accuracy', 'macro_f1')})
if len(CLASSES) == 2:
    baseline_library = library_size_baseline(train_records, test_records, CLASSES)
    print({k: baseline_library[k] for k in ('baseline', 'rule', 'train_accuracy', 'accuracy', 'macro_f1', 'auroc')})
else:
    baseline_library = None
    print('library-size baseline is defined for binary tasks only; skipped for', len(CLASSES), 'classes')"""
)

_ADAPT_CODE = _py(
    """import time

EPOCHS = 4  # @param {type:"integer"}
LEARNING_RATE = 2e-4  # @param {type:"number"}
BATCH_SIZE = 4  # @param {type:"integer"}
TRAINABLE_LAYERS = 2  # @param {type:"integer"}

started = time.perf_counter()
adapt_result = pipe.adapt(
    train_records,
    val_records,
    classes=CLASSES,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    trainable_layers=TRAINABLE_LAYERS,
    seed=SEED,
)
adapt_seconds = round(time.perf_counter() - started, 1)
print({'method': adapt_result['method'], 'trainable_parameters': adapt_result['trainable_parameters'], 'total_parameters': adapt_result['total_parameters'], 'precision': adapt_result['precision'], 'device': pipe.device, 'seconds': adapt_seconds})
for step in adapt_result['history']:
    print(step)"""
)

_EVAL_CODE = _py(
    """val_metrics = pipe.evaluate(val_records)
test_metrics = pipe.evaluate(test_records)
print({'split': 'validation', **{k: val_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}})
print({'split': 'test', **{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}})
for cls_name, row in test_metrics['per_class'].items():
    print({'class': cls_name, **row})

evaluation_report = {
    'task': 'single-cell state classification (bounded fine-tuning of Geneformer V2-104M)',
    'evidence': 'tutorial sample-sanity metrics on one stratified holdout; not a benchmark and not biology',
    'estimation': 'single train/validation/test split, seed ' + str(SEED) + ', no dispersion estimate',
    'data_source': data_source,
    'dataset_digest': dataset_manifest['digest'],
    'classes': CLASSES,
    'splits': {'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)},
    'baselines': {'majority': baseline_majority, 'library_size': baseline_library},
    'validation_metrics': val_metrics,
    'test_metrics': test_metrics,
    'delta_vs_majority': {k: round(test_metrics[k] - baseline_majority[k], 4) for k in ('accuracy', 'macro_f1')},
    'adaptation': {k: v for k, v in adapt_result.items() if k != 'trainable_parameter_names'},
    'adaptation_seconds': adapt_seconds,
}
with open('outputs/@STEM@_evaluation_report.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation_report, f, indent=2)
print({'delta_vs_majority': evaluation_report['delta_vs_majority'], 'report': 'outputs/@STEM@_evaluation_report.json'})

# One row per Section 8 run in this session, so a changed setting is read next to the default run (Section 13).
run_history = globals().get('run_history', [])
run_history.append({'run': len(run_history) + 1, 'data': 'BYOD' if USE_BYOD else 'sample', 'epochs': EPOCHS, 'trainable_layers': TRAINABLE_LAYERS, 'learning_rate': LEARNING_RATE, 'trainable_parameters': adapt_result['trainable_parameters'], 'final_val_accuracy': adapt_result['history'][-1].get('val_accuracy'), 'test_accuracy': test_metrics['accuracy'], 'test_macro_f1': test_metrics['macro_f1'], 'test_auroc': test_metrics['auroc']})"""
)

_INFER_CODE = _py(
    """if USE_BYOD:
    new_records = test_records[:6]
    new_source = 'first six BYOD test-split cells'
else:
    new_records = generate_sample_dataset(vocabulary, seed=7, size=6)
    new_source = 'freshly generated cells (seed 7)'
input_manifest = validate_inputs([r['counts'] for r in new_records], vocabulary, names=[r['id'] for r in new_records])
print({'new_source': new_source, 'verdict': input_manifest['verdict'], 'n_cells': input_manifest['n_cells'], 'max_tokens_observed': input_manifest['max_tokens_observed']})
inference_result = pipe.classify([r['counts'] for r in new_records], names=[r['id'] for r in new_records])
predictions = inference_result['predictions']
print({'decision_rule': inference_result['decision_rule']})
n_match = 0
for p, r in zip(predictions, new_records):
    n_match += p['label'] == r['label']
    print({'id': p['id'], 'predicted': p['label'], 'score': round(p['score'], 4), 'true_label': r['label']})
print({'matches': n_match, 'of': len(new_records), 'note': 'sanity check on generated labels, not an evaluation'})

with open('outputs/@STEM@_predictions.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'tokens_kept', 'predicted_label', 'score'] + [f'score_{c}' for c in CLASSES])
    for p in predictions:
        writer.writerow([p['id'], p['tokens_kept'], p['label'], f"{p['score']:.6f}"] + [f"{p['scores'][c]:.6f}" for c in CLASSES])
print('wrote outputs/@STEM@_predictions.csv')"""
)

_EXPORT_CODE = _py(
    """artifact_dir = Path('outputs/@STEM@_adapter')
pipe.save_artifact(artifact_dir, metadata={'data_source': data_source, 'dataset_digest': dataset_manifest['digest'], 'test_metrics': {k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}})
with open(artifact_dir / ARTIFACT_MANIFEST_NAME, encoding='utf-8') as f:
    artifact_manifest = json.load(f)
print({'format': artifact_manifest['format'], 'base_model': artifact_manifest['base_model'], 'classes': artifact_manifest['classes'], 'epochs': artifact_manifest['adaptation']['epochs'], 'trainable_layers': artifact_manifest['adaptation']['trainable_layers'], 'n_tensors': len(artifact_manifest['tensors']), 'files': artifact_manifest['files']})

reloaded_pipe = GeneformerPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR)
reloaded_result = reloaded_pipe.classify([r['counts'] for r in new_records], names=[r['id'] for r in new_records])
max_score_diff = 0.0
for before, after in zip(predictions, reloaded_result['predictions']):
    assert before['id'] == after['id'] and before['label'] == after['label'], f'reload parity failure on {before["id"]}'
    max_score_diff = max(max_score_diff, abs(before['score'] - after['score']))
assert max_score_diff < 1e-5, f'reload score drift {max_score_diff}'
print({'reload_parity': 'PASS', 'labels_equal': True, 'max_abs_score_diff': max_score_diff, 'loaded_from': reloaded_pipe.adaptation.get('loaded_from_artifact')})
del reloaded_pipe"""
)

_RESULT_CODE = _py(
    """import platform

result_payload = {
    'task': 'single-cell state classification adaptation (Geneformer V2-104M)',
    'pipeline_class': 'GeneformerPipeline',
    'model_id': MODEL_ID,
    'model_revision': MODEL_REVISION,
    'model_subfolder': MODEL_SUBFOLDER,
    'model_license': MODEL_LICENSE,
    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],
    'notebook_source': NOTEBOOK_SOURCE,
    'data_source': data_source,
    'dataset_manifest': dataset_manifest,
    'vocabulary': {'tokens': len(vocabulary.tokens), 'corpus_medians': len(vocabulary.medians), 'special_tokens': vocabulary.special},
    'encoding_example': {'id': example_a['id'], 'label': example_a['label'], 'top_ranked_gene_ids': encoded_a['ranked_gene_ids'][:8], 'tokens_kept': encoded_a['n_kept'], 'truncated': encoded_a['n_truncated']},
    'embedding_summary': {'n_cells': embedding_result['n_cells'], 'dimension': embedding_result['dimension'], 'pooling': embedding_result['pooling']},
    'evaluation_report': evaluation_report,
    'inference': {'new_source': new_source, 'decision_rule': inference_result['decision_rule'], 'predictions': predictions},
    'artifact_format': ARTIFACT_FORMAT,
    'artifact_format_version': ARTIFACT_FORMAT_VERSION,
    'artifact_manifest': artifact_manifest,
    'reload_parity': {'labels_equal': True, 'max_abs_score_diff': max_score_diff},
    'run_history': run_history,
    'runtime': {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'safetensors': safetensors.__version__,
        'device': pipe.device,
        'precision': 'float32',
    },
}
with open('outputs/@STEM@_result.json', 'w', encoding='utf-8') as f:
    json.dump(result_payload, f, indent=2)
# Every export of this run describes one model: the report, the artifact and this file.
assert evaluation_report['adaptation']['epochs'] == artifact_manifest['adaptation']['epochs'] == EPOCHS
assert evaluation_report['adaptation']['trainable_layers'] == artifact_manifest['adaptation']['trainable_layers'] == TRAINABLE_LAYERS

print('outputs/:')
for path in sorted(Path('outputs').rglob('*')):
    if path.is_file():
        print(f'  - {path.as_posix()} ({path.stat().st_size / 1024:.1f} KB)')"""
)

_ACTIVITY_CODE = _py(
    """# Section 9 adds one row per Section 8 run in this session; this cell only prints them.
columns = ['run', 'data', 'epochs', 'trainable_layers', 'learning_rate', 'trainable_parameters', 'final_val_accuracy', 'test_accuracy', 'test_macro_f1', 'test_auroc']
print(' | '.join(columns))
for row in run_history:
    print(' | '.join(str(row[column]) for column in columns))
if len(run_history) == 1:
    print('One run so far. Set EPOCHS = 2 in Section 8, select that cell and choose Runtime > Run after; this table then shows both runs.')"""
)

TEMPLATE = {
    "package": "geneformer_single_cell_pipeline",
    "repo_name": REPO,
    "stem": "geneformer_single_cell",
    "notebook_name": "geneformer_single_cell_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "infrastructure_labels": True,
    "collapse_model_cell": True,
    "isolated_runtime": True,
    # The fleet's uv isolated-environment mechanism (ast-audio-classification-pipeline / bioclip2-biodiversity-pipeline):
    # managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`. The pins equal ast-audio-classification-pipeline's,
    # so the lock is that repository's (16eee39) with the requesting project renamed.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (the "
        "kernel's own packages are left alone, so no restart is needed), stages and digest-verifies the pinned Geneformer "
        "V2-104M snapshot (6 files, ~421 MB: the checkpoint plus the three gene dictionaries), loads those dictionaries "
        "through a restricted unpickler, generates a deterministic 64-cell tutorial dataset over real human Ensembl gene ids "
        "(no download), validates it against the cell contract, splits it into stratified train/validation/test sets and "
        "checks each split, rank-value encodes cells and inspects one encoding, computes cell embeddings, measures a "
        "majority-class and a library-size baseline on the test split, runs a bounded AdamW fine-tuning of the cell-state "
        "classification head and the last two encoder layers, evaluates accuracy, macro-F1 and AUROC on the held-out test "
        "split, classifies six freshly generated cells, exports the adapter as safetensors with a manifest, and reloads that "
        "artifact into a fresh pipeline to verify prediction parity. The default path needs no repository clone, no DIMER "
        "worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.2 §5). Measured times: "
        "a Kaggle Tesla T4 run took 279.4 s of cell "
        "time (2026-09-18, including a pinned in-kernel install this version no longer does); a local Windows CPU check of "
        "this version's learner cells took 90.8 s with the install skipped and the files pre-staged (2026-10-04), 65.9 s of it "
        "in the fine-tuning cell. Building the isolated environment (PyTorch "
        "with its CUDA libraries) and downloading the checkpoint come on top and usually take a few minutes (an estimate; no "
        "hosted run of this version is recorded yet)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4, select that cell and choose "
        "**Runtime → Run after** to supply your own labelled cells as a genes-as-columns CSV (`id`, one column per gene, "
        "`label`), a JSON array or a JSONL file of id/counts/label records — through the upload dialog on Colab, or as a path "
        "in `BYOD_PATH` on any runtime. It passes through the same validation, stratified split, baselines, rank-value "
        "encoding, adaptation, held-out evaluation, inference, artifact export and reload-parity cells as the synthetic "
        "sample; Section 4 first drops the head trained on the sample, so nothing trained on the sample can be evaluated or "
        "exported under your file's name. The expected schema, the identifier requirements, the minimum size and the "
        "ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is "
        "optional and never part of the default path."
    ),
    "pipeline_class": "GeneformerPipeline",
    "weights_key": "geneformer-v2-104m",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers", "safetensors"],
    "title": "Geneformer V2-104M — DIMER E2E single-cell state classification tutorial (standalone)",
    "badges": BADGES,
    "capability": "rank-value encoding of single-cell transcriptomes, cell embeddings, and bounded cell-state classification fine-tuning",
    "intro": (
        "Geneformer is a transformer pretrained on single-cell transcriptomes, and its central idea is the input format rather than the "
        "architecture. A cell is not fed to the model as a vector of counts: each gene's expression is divided by that gene's median "
        "expression across the pretraining corpus, the genes are **ranked** by the result, and the ranked gene identifiers become the "
        "token sequence. Housekeeping genes that are high in every cell are scaled down; a transcription factor that is lowly expressed "
        "but rarely expressed at all moves up. The consequence is that sequencing depth and absolute count scale largely drop out, and "
        "what the model reads is the order of genes within the cell. The pinned V2-104M checkpoint is a 12-layer BERT encoder with a "
        "20,275-token gene vocabulary and a 4,096-token input, pretrained on ~104 million human transcriptomes.\n\n"
        "This tutorial implements that encoding in the notebook, from the pinned gene dictionaries, and then adapts the encoder to a "
        "cell-state classification task. The tutorial dataset is synthetic but uses real human Ensembl ids, and it is built so that "
        "library size carries no signal: every cell is scaled to the same total counts and detects the same genes, and the two classes "
        "differ only in which of two median-matched gene programmes ranks above the other. A baseline that thresholds total counts "
        "therefore cannot separate them — which is exactly the failure mode rank-value encoding exists to avoid.\n\n"
        "**Who this is for.** A learner who knows basic Python, has met a single-cell count matrix (cells × genes) and the idea of a "
        "train/validation/test split, and wants to see how a pretrained single-cell foundation model reads a cell and how it is "
        "fine-tuned and measured honestly: baselines first, a bounded fine-tune, one look at the held-out split, an exported adapter "
        "that reloads. No prior experience with Geneformer, transformers or fine-tuning is assumed; each term is explained where it is "
        "first used and again in the **Glossary** at the end. No GPU is required (CPU works; a GPU is used automatically).\n\n"
        "**Input → Model → Output.**\n\n"
        "| | Embedding (Section 6) | Classification (Sections 7–11) |\n"
        "|---|---|---|\n"
        "| Input | one cell's gene counts → **rank-value tokens** (genes ranked by count ÷ corpus median, at most 4,094 genes) | labelled cells `id`, `counts`, `label`, split into training, validation and test |\n"
        "| Model | Geneformer V2-104M encoder, token states mean-pooled; no head | the same encoder + a new classification head, the head, the pooler and the last `TRAINABLE_LAYERS` encoder layers trained |\n"
        "| Output | one 768-d vector per cell; no label, no score | held-out accuracy, macro-F1 and AUROC beside two baselines, argmax labels with per-class scores, and a safetensors adapter that reloads with identical predictions |\n\n"
        "**How to use this notebook.** Choose a runtime (Colab, Kaggle or Linux Jupyter; a GPU is faster, CPU works), then "
        "**Runtime → Run all**. Sections 1–3 are **infrastructure** — the isolated environment, the carried code and the model "
        "verification — and can be run without study; their code is collapsed. The learning path starts in Section 4. Form "
        "fields (`# @param`) are the only values meant to be edited; the defaults reproduce the default path. Each learner "
        "section asks you to **predict** before it runs; the next section opens with **What to notice** and a collapsible "
        "**Check your reasoning** block with a worked answer. Each optional experiment names the field to change and the cell to "
        "re-run from (select that cell, then **Runtime → Run after**). Section 13 is a **change-one-thing activity**. "
        "**Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Your notes are optional.\n\n"
        "**Roadmap:** *core concepts* — 4 gene dictionaries and the dataset → 5 reading a rank-value encoding → 6 embeddings as "
        "representations; *evaluation practice* — 7 baselines → 8 bounded fine-tuning → 9 the held-out evaluation; "
        "*engineering* — 10 inference on new cells → 11 export and reload → 12 provenance → 13 **change one thing: train for fewer "
        "epochs** → conclude."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) explain how a cell's counts become a rank-value token sequence and say what the "
        "encoding drops (Section 5); (2) say why the cosine between two mean-pooled cell embeddings is an inspection, not an "
        "evaluation (Section 6); (3) predict what a library-size baseline scores when every cell has the same total counts, and say "
        "what it rules out (Section 7); (4) state which parameters a bounded fine-tuning trains and why validation is monitoring only "
        "here (Section 8); (5) interpret accuracy, macro-F1 and AUROC on a 16-cell test split and judge what a perfect score does and "
        "does not show (Section 9); (6) check that an exported adapter reproduces the evaluated model (Section 11); and (7) predict, "
        "measure and explain how accuracy and AUROC diverge for an under-trained head (Section 13)."
    ),
    "exclusions": (
        "in-silico perturbation, gene classification, multitask or continual learning, masked-gene prediction, the V1-10M / "
        "V2-316M / cancer-tuned checkpoints, and reading `.h5ad`, `.loom` or other single-cell file formats. The repository "
        "exposes none of these; its input is a mapping of gene identifier to count."
    ),
    "prerequisites": [
        "- **Learner:** basic Python; what a single-cell count matrix is and why library size (total counts) differs between cells; what a train/validation/test split protects against. The notebook explains rank-value encoding, corpus medians, embeddings, mean pooling, encoder layers, the pooler, the classification head, fine-tuning, AdamW, accuracy, macro-F1, AUROC, softmax scores, the adapter and reload parity where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab, Kaggle or Linux Jupyter. Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter, and a Windows or macOS kernel is not supported (Section 1 stops with that message). CPU is enough — the default fine-tuning takes a few minutes — and CUDA is used automatically when present. The locked install (PyTorch 2.14.0 with its CUDA libraries) and the ~421 MB checkpoint are the large downloads.",
        "- **Data contract:** records are `{id, counts, label}` where `counts` maps a human Ensembl gene id (`ENSG...`) or an approved gene symbol to a non-negative count. At least 10 genes must be detected per cell and carry both a Geneformer token and a corpus median; at most 4,094 ranked genes fit the input; 2..20 classes, unique ids, at most 2,000 records. **Minimum size:** at least 8 records and 3 per class. That is enough: the stratified split puts every class at least once into each of training, validation and test (with the default 20 % / 25 % fractions), which is all the fine-tuning and evaluation cells require of a split; Section 4 checks every split before any model runs, and a refusal names the split and the number of records a class needs.",
        "- **BYOD file (Section 4):** a genes-as-columns CSV (`id`, one column per gene, `label`), a JSON array of objects, or JSONL, saved as UTF-8 (a spreadsheet's 'CSV UTF-8'). `.h5ad`, `.loom` and matrix-market files are refused with a message: export the cells you want to a CSV first. On Colab the upload dialog opens; on any runtime set `BYOD_PATH` to the file instead. Your records are written to `outputs/geneformer_single_cell_byod_dataset.csv`; the sample is written to `outputs/geneformer_single_cell_sample_dataset.csv`, which is also a template of the expected shape.",
        "- **Species:** the pinned vocabulary is human. Mouse or other non-human identifiers will not resolve, and the validation stage says so rather than silently dropping them.",
        "- **Privacy:** cells you bring are your responsibility. Do not upload confidential or restricted data — patient-derived or unpublished cells included — to a hosted runtime unless you are authorized to process it there. The default path uploads nothing and the BYOD branch keeps your file inside this runtime.",
        "- **Expected log lines:** loading `BertForSequenceClassification` prints that the classifier head and pooler weights are newly initialised — that is the head this tutorial trains, not a defect.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Gene dictionaries, sample cells, validation and split\n\n"
                "Geneformer's tokenizer is not a vocabulary file: it is three dictionaries shipped with the model — gene id to token, "
                "gene id to **corpus median** expression (the gene's median across the pretraining cells), and gene symbol to gene id. "
                "They are Python pickles, so this pipeline loads them through `RestrictedUnpickler`, which refuses every global except "
                "the numpy scalar types the median dictionary legitimately contains; an unexpected class raises instead of executing "
                "(NOTEBOOK_SPEC 2.2 §20). They have already been digest-verified as part of the snapshot in Section 3.\n\n"
                "The default dataset is then generated in code with a fixed seed: 32 cells per class over real human Ensembl ids, each "
                "scaled to the same library size. `validate_dataset` checks the schema, the count values, the per-cell gene coverage and "
                "the class coverage before any model runs, and — because it is given the vocabulary — also reports how many genes per "
                "cell can actually be encoded. `split_dataset` shuffles **within each class** and cuts 20 % validation / 25 % test, and "
                "`validate_splits` checks that each split holds every class (the minimum the fine-tuning and evaluation cells apply; a "
                "refusal names the split). Random splitting assumes the cells are independent, which holds for generated cells and not "
                "for cells from the same donor or batch (see Interpretation and limits). The cell first calls `pipe.reset_adaptation()`: "
                "a new dataset needs a new head, so any head an earlier run trained is dropped and no later cell can evaluate or export "
                "it under this dataset's name.\n\n"
                "With `USE_BYOD = True`, your file (upload dialog on Colab, or `BYOD_PATH` on any runtime) goes through "
                "`load_byod_dataset` instead; see the Prerequisites for the format and the minimum size.\n\n"
                "**Predict before running:** the test split will hold 16 cells, 8 per class. By how many percentage points does one cell "
                "moved from wrong to right change the test accuracy?"
            ),
            "code": _DATA_CODE,
        },
        {
            "md": (
                "**What to notice:** 20,275 tokens and 42,005 corpus medians loaded; 64 cells, classes `['programme-A', 'programme-B']`; "
                "every cell with the same library size and the same detected genes; splits 36 / 12 / 16 with both classes in each; and the "
                "written `outputs/{stem}_sample_dataset.csv` — the exact file shape BYOD expects.\n\n"
                "<details><summary>Check your reasoning</summary>6.25 points: one cell is 1/16 of the test split. Keep that in mind in "
                "Section 9 — a difference of one or two cells between two models is within what one seeded split of 16 can show.</details>\n\n"
                "## 5. Read one rank-value encoding\n\n"
                "This is the cell to slow down on. `rank_value_encode` takes one cell's counts, normalises them to a fixed library size, "
                "divides each gene by its corpus median, ranks the genes by that ratio and maps them to tokens — a **rank-value encoding**. "
                "The printout compares the top of the ranking for one cell of each class.\n\n"
                "The encoding also reports what it dropped — genes with a zero count, genes with no Geneformer token, genes with no corpus "
                "median — and how many ranked genes were truncated at the 4,094-token input limit. Nothing is dropped silently (VAL7).\n\n"
                "**Predict before running:** the two cells detect exactly the same genes and have the same total counts. Will their eight "
                "top-ranked genes be the same?"
            ),
            "code": _ENCODE_CODE,
        },
        {
            "md": (
                "**What to notice:** the same `detected` and `library_size` for both cells, `same genes detected in both cells: True`, "
                "zero unknown genes and zero truncated — and two different top-8 lists.\n\n"
                "<details><summary>Check your reasoning</summary>No. Ranking divides each count by the gene's corpus median before sorting, "
                "so what decides the top of the list is how far a gene sits **above its usual level**, not its raw count. The generator puts "
                "the expressing programme's genes at about 4× their median and the other programme's at about 0.25×, so each class puts its "
                "own programme at the head of the ranking. The library size and the gene set are identical; only the order differs — and the "
                "order is all the model reads.</details>\n\n"
                "## 6. Cell embeddings (representation, not prediction)\n\n"
                "`pipe.embed` runs the verified encoder over the rank-value tokens and returns one 768-dimensional vector per cell: the "
                "mean of the last hidden state over the cell's tokens, padding excluded — **mean pooling**. Embeddings are "
                "representations — they carry no label and no metric of their own; a downstream labelled task is what gives them meaning "
                "(EVAL9). The cell embeds eight validation cells, writes them with their ids to `outputs/{stem}_embeddings.csv` (OUT4), "
                "and prints the mean **cosine similarity** (1.0 = same direction) within and between classes as an inspection, not an "
                "evaluation.\n\n"
                "**Predict before running:** before any training, will two cells of the same class have clearly more similar embeddings "
                "than two cells of different classes?"
            ),
            "code": _EMBED_CODE,
        },
        {
            "md": (
                "**What to notice:** 8 cells, 768 dimensions, the pooling description, and two mean cosines that are both very close to 1 "
                "and very close to each other. That is the expected shape, not a failure: read the gap between them, not their size.\n\n"
                "<details><summary>Check your reasoning</summary>No — not clearly. Every cell here detects the same ~350 genes, and mean "
                "pooling averages the encoder's state over all of their tokens, so all cells point in almost the same direction and every "
                "cosine is close to 1. The two classes differ only in the order of 48 programme genes, a small part of each average. In the "
                "local CPU check of this notebook's code (pinned versions, 2026-10-04) the within-class mean was 0.9961 and the "
                "between-class mean 0.9948: a gap of about 0.001, on 8 cells, with no label used. It neither shows that the frozen "
                "embeddings separate the classes nor that they cannot; separability is measured with labels, after training, in Section 9 — "
                "which is why the next sections train a head.</details>\n\n"
                "## 7. Baselines on the test split\n\n"
                "Two trivial predictors set the floor before any training (EVAL10/EVAL11), each fitted on the training split only (SPL8) "
                "and scored on the test split. `majority_baseline` predicts the most frequent training class for every test cell — 0.5 "
                "accuracy on a balanced split. `library_size_baseline` fits one threshold on total counts per cell; library size is the "
                "first confounder to rule out in any single-cell classification. A fine-tuned model that clears both has learned something "
                "about the gene ranking rather than about sequencing depth. The library-size baseline is binary only and is skipped for a "
                "BYOD dataset with more classes.\n\n"
                "**Predict before running:** what test accuracy will the library-size baseline reach on this dataset, and why?"
            ),
            "code": _BASELINE_CODE,
        },
        {
            "md": (
                "**What to notice:** the majority baseline's accuracy and macro-F1, and the library-size baseline's rule, train accuracy, "
                "test accuracy and AUROC.\n\n"
                "<details><summary>Check your reasoning</summary>About chance. Every cell was scaled to the same library size (the totals "
                "differ only by rounding), so a threshold on total counts has nothing to work with: in the local CPU check it scored 0.5 "
                "accuracy with AUROC 0.49, the same as the majority baseline. On real data, run this baseline first — if it already scores "
                "well, your labels may be predictable from sequencing depth alone.</details>\n\n"
                "## 8. Bounded fine-tuning\n\n"
                "`pipe.adapt` drops any earlier head, builds `BertForSequenceClassification` from the verified checkpoint (the head and the "
                "**pooler** — the small layer that summarises the first token — are newly initialised; the log line says so), **freezes** "
                "every parameter except the head, the pooler and the last `TRAINABLE_LAYERS` encoder layers (each **encoder layer** is one "
                "transformer block of the 12), and runs **AdamW** (the optimiser) with the hyperparameters below (FT4/FT6): tutorial values "
                "chosen for a few minutes of CPU, not production settings. Validation metrics are computed after each **epoch** (one pass "
                "over the training split) for **monitoring only**; the final epoch's weights are kept, so no selection happens on the "
                "validation split (EVAL14). Training loss going down is optimisation evidence, not task-quality evidence (FT7) — Section 9 "
                "is where quality is measured. Every run starts again from the pretrained checkpoint, so a rerun with other settings never "
                "builds on an earlier one.\n\n"
                "**Predict before running:** about 14.8 M of 104 M parameters are trained. In which of the four epochs will validation "
                "accuracy first leave 0.5?"
            ),
            "code": _ADAPT_CODE,
        },
        {
            "md": (
                "**What to notice:** the trainable/total parameter counts, the train loss per epoch and the validation accuracy per epoch.\n\n"
                "<details><summary>Check your reasoning</summary>In the local CPU check 14,767,874 of 104,366,594 parameters were trained and "
                "validation accuracy went 0.5 → 0.5 → 1.0 → 1.0: it stays at chance for two epochs while the new head finds the direction, "
                "then jumps. Your run may jump one epoch earlier or later on another device; what matters is that it leaves 0.5 before the "
                "last epoch. If it is still at 0.5 at the end, the head is under-trained, not broken — Section 13 shows what that looks "
                "like.</details>\n\n"
                "## 9. Held-out evaluation\n\n"
                "`pipe.evaluate` classifies every cell of a split and reports `accuracy` (discrete correctness under the argmax rule), "
                "`macro_f1` (the unweighted mean of per-class F1, which exposes a model that ignores a class), per-class precision/recall/F1 "
                "with support, and `auroc` (ranking quality of the positive-class score, independent of the argmax threshold). The **test "
                "split** was never used for training or monitoring, so its numbers are the independent evidence (SPL6/SPL7). These are "
                "tutorial metrics on a synthetic 16-cell split (EVAL6): one holdout, no dispersion estimate. The report, with both baselines "
                "and the deltas against them, is written to `outputs/{stem}_evaluation_report.json`. The cell also adds this run to "
                "`run_history`, which Section 13 prints.\n\n"
                "**Predict before running:** what test accuracy, macro-F1 and AUROC do you expect, and what would a perfect score here tell "
                "you about Geneformer?"
            ),
            "code": _EVAL_CODE,
        },
        {
            "md": (
                "**What to notice:** the validation and test rows, the per-class rows with their support, and `delta_vs_majority`.\n\n"
                "<details><summary>Check your reasoning</summary>In the local CPU check and in the recorded Kaggle T4 run of the previous "
                "version the test split scored accuracy, macro-F1 and AUROC 1.0 (n = 16) — 0.5 above both baselines. That tells you the "
                "rank-value encoding carries the class signal and the bounded adaptation picks it up. It does not tell you that Geneformer "
                "predicts any real cell state: the classes are a generator rule, 16 synthetic cells give no dispersion estimate, and a "
                "different seed or a GPU could move one cell (6.25 points).</details>\n\n"
                "## 10. Inference on new cells\n\n"
                "`pipe.classify` returns, per cell, the argmax `label`, its `score` and the full `scores` dictionary in class order. The "
                "scores are **softmax** outputs of a head trained on a few dozen cells — **not calibrated probabilities** (UNC2); the only "
                "decision rule is argmax (UNC3), and a deployment that must trade false positives against false negatives owns its own "
                "threshold. On the default path the new cells are generated with a different seed, so their true labels are known and shown "
                "as a check; on the BYOD path the first six test-split cells stand in as new data (INF2). Predictions are written to "
                "`outputs/{stem}_predictions.csv` with ids and per-class scores.\n\n"
                "**Predict before running:** will the six scores be close to 1.0, and does a score of 0.99 mean the label is 99 % likely "
                "to be right?"
            ),
            "code": _INFER_CODE,
        },
        {
            "md": (
                "**What to notice:** six predictions with their true labels, the scores, and `matches`.\n\n"
                "<details><summary>Check your reasoning</summary>In the local CPU check all 6 were right with scores of about 0.998. That "
                "is a confident head on an easy synthetic rule, not a 99.8 % probability of being right: softmax scores are not calibrated, "
                "and six cells drawn from the same generator are a sanity check, not an evaluation.</details>\n\n"
                "## 11. Export the adapter and verify a fresh reload\n\n"
                "`pipe.save_artifact` writes only the trained tensors (head, pooler and the unfrozen encoder layers) as "
                "`adapter.safetensors` — **safetensors** is a file format that stores tensors without executable code — plus a "
                "`manifest.json` that records the artifact format, the exact base model id, revision **and subfolder** the tensors belong "
                "to (ART4 — the repository hosts several checkpoints, so the subfolder is part of the identity), the class order, the "
                "tensor names, the file size and SHA-256, and the adaptation configuration (OUT8). `GeneformerPipeline.from_artifact` then "
                "re-verifies the base snapshot, checks the artifact manifest and digests **before** deserialising, rebuilds the classifier "
                "and overlays the tensors — a fresh object from files, not the in-memory model (VER2). The cell compares its predictions on "
                "the same new cells with the pre-export ones: labels must match exactly and scores within `1e-5` (VER4).\n\n"
                "**Predict before running:** will the reloaded model's scores be identical to the in-memory ones, or only close?"
            ),
            "code": _EXPORT_CODE,
        },
        {
            "md": (
                "**What to notice:** the manifest summary (classes, `epochs`, `trainable_layers`, tensor count, file size and SHA-256), "
                "`reload_parity: PASS` and `max_abs_score_diff`.\n\n"
                "<details><summary>Check your reasoning</summary>Identical on the same device: the local CPU check measured a maximum "
                "difference of 0.0 (36 tensors, about 59 MB). On a GPU a difference in the last digits is possible, which is why the check "
                "allows `1e-5`. Loading a file is not reproducing a result; the parity check is what shows the export is the model that was "
                "evaluated.</details>\n\n"
                "## 12. Result export and provenance\n\n"
                "The last output, `outputs/{stem}_result.json`, gathers everything a reader needs to interpret the files above: the "
                "notebook source revision and module digests, the model id, immutable revision, subfolder and licence, the dataset source "
                "and digest, the adaptation configuration, baseline and held-out metrics, the new-cell predictions, the artifact manifest, "
                "the reload-parity result, the run history, and the runtime versions and device (OUT6/OUT7). The cell asserts that the "
                "report, the artifact and this file describe the same model. No credential is involved anywhere in this notebook, so none "
                "can leak into it (OUT10).\n\n"
                "**Predict before running:** how many files will `outputs/` hold after a default run?"
            ),
            "code": _RESULT_CODE,
        },
        {
            "md": (
                "**What to notice:** seven files — the sample CSV, the embeddings, the evaluation report, the predictions and the result "
                "file at the top level, and `adapter.safetensors` plus `manifest.json` in the adapter folder.\n\n"
                "<details><summary>Check your reasoning</summary>Seven: five files at the top level and two inside "
                "`outputs/geneformer_single_cell_adapter/` (the local CPU check listed exactly these). After a BYOD run "
                "`geneformer_single_cell_byod_dataset.csv` is added, and a rerun overwrites the others, so they always describe the latest "
                "run.</details>\n\n"
                "## 13. Your turn — change one thing: train for fewer epochs\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.**\n\n"
                "1. **Predict:** with `EPOCHS = 2` the head stops training at the point where, in Section 8, validation accuracy was still "
                "0.5. What will the test **accuracy** be? And the test **AUROC**? Write both guesses down — they need not move together.\n"
                "2. **Change one thing:** in Section 8 set `EPOCHS = 2` and nothing else.\n"
                "3. **Run:** select the Section 8 cell and choose **Runtime → Run after** (it re-runs Sections 8–13, so the report, the "
                "predictions, the adapter and the result file are all rewritten for this run and describe one model). `adapt` starts again "
                "from the pretrained checkpoint.\n"
                "4. **Observe:** this cell prints one row per Section 8 run in this session — the setting, the trained parameter count, the "
                "final validation accuracy and the test accuracy, macro-F1 and AUROC.\n"
                "5. **Explain:** why can accuracy and AUROC disagree, and which one tells you the encoding carries the signal?\n\n"
                "<details><summary>Check your reasoning</summary>In the local CPU check (and in the review's CPU check of the previous "
                "version) `EPOCHS = 2` gave test accuracy 0.5 with AUROC 1.0. AUROC 1.0 means every `programme-B` cell already gets a higher "
                "score than every `programme-A` cell — the ranking is right — but the argmax cut has not moved yet, so every cell gets the "
                "same label and accuracy sits at chance. Ranking comes before thresholding: AUROC shows the signal is there, accuracy shows "
                "whether the decision rule uses it. On a GPU the scores can differ in the last digits, so a cell or two may already cross "
                "the cut; the point stands while AUROC stays near 1 and accuracy stays well below it. Other experiments in the same pattern: "
                "`TRAINABLE_LAYERS = 0` (head and pooler only), `LEARNING_RATE = 5e-5` — each adds a row.</details>"
            ),
            "code": _ACTIVITY_CODE,
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The fine-tuned head separates two classes of cells that share their library size, detect the same genes and differ only in which "
        "median-matched gene programme ranks higher — which the library-size baseline cannot do. That is the whole claim of this notebook: "
        "the rank-value encoding carries the signal, and a bounded adaptation can pick it up. The test split has 16 synthetic cells, the "
        "metrics come from one seeded holdout with no dispersion estimate, and the classes are defined by a generator rule rather than by "
        "biology — so a perfect score here says the adaptation contract works, not that Geneformer predicts any real cell state.\n\n"
        "On real data the same workflow needs more care than this sample shows. Cells from one donor, plate, or 10x run are not "
        "independent, so a random split leaks and a donor- or batch-level split is required. Labels transferred from a reference atlas "
        "carry that atlas's errors. Ambient RNA, doublets and dying cells change the ranking before the model sees it, and none of that is "
        "detected here. The pinned vocabulary is human; other species will not resolve. The softmax scores are uncalibrated, and the "
        "embeddings are representations that need a labelled downstream task before any quality can be stated.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model and its gene dictionaries, load those dictionaries without executing arbitrary code, "
        "rank-value encode cells, execute bounded fine-tuning, evaluate against trivial baselines on an independent split, and emit the "
        "shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark superiority, "
        "production fitness, or biological validity of the classes.\n\n"
        "**Next experiments** (each starts after the default Run all; every adaptation starts again from the pretrained checkpoint):\n\n"
        "1. **Fewer epochs:** the Section 13 activity (`EPOCHS = 2`), **Run after** from Section 8.\n"
        "2. **Head only:** in Section 8 set `TRAINABLE_LAYERS = 0` (only the head and the pooler are trained), then **Run after** from "
        "Section 8, and compare the trainable parameter count and the test metrics in the new Section 13 row. Predict first: does this "
        "task need the encoder layers to change?\n"
        "3. **Another split:** in Section 4 change `SEED`, then **Run after** from Section 4; the baselines and the fine-tuning are "
        "recomputed on the new split.\n"
        "4. **Your own labelled cells:** in Section 4 set `USE_BYOD = True` (and upload, or set `BYOD_PATH`), then **Run after** from "
        "Section 4. Read the library-size baseline first — if it already scores well, your labels may be predictable from sequencing "
        "depth alone.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment is built from manylinux wheels; use Colab, "
        "Kaggle or a Linux Jupyter server.\n"
        "- **Section 1 fails to download `uv`, Python or a package.** The runtime needs `pypi.org`, `files.pythonhosted.org` and the "
        "python-build-standalone release host. Re-run the cell; a size or SHA-256 mismatch is refused on purpose.\n"
        "- **\"The isolated environment's Python process exited\".** Usually out of memory. Restart the session and choose **Run all** "
        "again; on a small CPU runtime lower `BATCH_SIZE` in Section 8.\n"
        "- **Section 3 reports a size or SHA-256 mismatch, or a download error.** A snapshot file was altered, truncated or not fetched; "
        "delete the `weights/geneformer-v2-104m` folder and run Section 3 again.\n"
        "- **Section 3 or 4: \"refusing to unpickle\" / a restricted-unpickler error.** A gene dictionary contains a class outside the "
        "allow-list. That is the safety check working: do not bypass it; re-run Section 3 so the pinned, digest-verified files are used.\n"
        "- **Section 4: \"Upload exactly one … file (got 0)\".** The upload was cancelled or empty. Run Section 4 again and pick one file, "
        "or set `BYOD_PATH`.\n"
        "- **Section 4: \"The upload dialog exists only on Google Colab\".** On Kaggle or Jupyter, put your file in the runtime and set "
        "`BYOD_PATH` to its path.\n"
        "- **Section 4: \"unsupported BYOD file type '.h5ad'\" or \"… is not UTF-8 text\".** Export the cells to a genes-as-columns CSV "
        "saved as UTF-8 and run Section 4 again.\n"
        "- **Section 4 refuses a BYOD dataset.** The message names the record and the rule (unknown gene ids — are they human Ensembl "
        "ids or symbols? — too few detected genes, a non-numeric count, a duplicate id, too few records or classes), or the split and "
        "the number of records a class needs. Fix the file and run Section 4 again.\n"
        "- **Section 9, 10 or 11: \"requires an adapted head\".** Section 8 has not completed since the dataset last changed; run Section "
        "8 (**Run after**) first.\n"
        "- **Section 11's parity assertion fails.** The export or reload is broken; run Sections 8–12 again. Do not use that artifact.\n"
        "- **Slow on CPU.** Each epoch trains on 36 cells of up to 4,096 tokens through the 12-layer encoder; a GPU runtime is several "
        "times faster. Lower `EPOCHS` for a quicker experiment.\n"
        "- **CUDA out of memory.** Lower `BATCH_SIZE` in Section 8, or switch the runtime to CPU.\n\n"
        "## Glossary\n\n"
        "- **Rank-value encoding:** a cell's genes sorted by count ÷ corpus median (highest first), turned into token ids; the order, "
        "not the counts, is what the model reads.\n"
        "- **Corpus median:** a gene's median normalised expression across the pretraining cells; dividing by it scales down genes that "
        "are high everywhere.\n"
        "- **Library size:** a cell's total counts; it reflects sequencing depth as much as biology, which is why it is the first "
        "baseline.\n"
        "- **Token / vocabulary:** one gene id per token; the pinned vocabulary has 20,275 tokens, human genes only.\n"
        "- **Embedding / mean pooling / cosine similarity:** a fixed-length vector for a whole cell / the average of the encoder's "
        "768-d token states / how closely two vectors point the same way (1.0 = identical direction).\n"
        "- **Encoder layer / pooler / classification head:** one of the 12 transformer blocks / a small layer that summarises the first "
        "token / a new layer mapping that summary to one score per class.\n"
        "- **Fine-tuning / frozen parameters:** continuing training of a pretrained model on labelled data; frozen parameters are left as "
        "they are. **AdamW** is the optimiser; an **epoch** is one pass over the training split.\n"
        "- **Accuracy / macro-F1:** the share of correct labels / the unweighted mean of per-class F1, which punishes a forgotten class.\n"
        "- **AUROC:** the probability that a random positive cell is scored above a random negative one; it measures ranking, not the "
        "argmax cut.\n"
        "- **Softmax score:** the model's per-class output scaled to sum to 1; not a calibrated probability.\n"
        "- **Baseline:** a trivial predictor scored on the same split; a model must beat it to have learned anything.\n"
        "- **Adapter / safetensors / reload parity:** the saved trained tensors / a code-free tensor file format / the check that base + "
        "adapter reproduces the in-memory model's predictions.\n"
        "- **Restricted unpickler:** a pickle loader that refuses every class outside an allow-list, so loading a dictionary cannot run "
        "code.\n"
        "- **Donor- or batch-level split:** keeping all cells of one donor or batch on one side of the split, so the test split measures "
        "generalisation to new donors rather than memory of familiar ones.\n\n"
        "## Conclusion (your notes)\n\n"
        "Fill in from the numbers this run printed; keep each claim to what the evidence shows.\n\n"
        "- **Task:** which cells, which classes, which split sizes?\n"
        "- **Principal result:** the fine-tuned test accuracy, macro-F1 and AUROC, with `n`.\n"
        "- **Baselines / reference:** the majority and library-size baselines on the same split, and `delta_vs_majority`.\n"
        "- **Uncertainty or failure mode:** how many cells is a difference? What did your Section 13 run change, and what did AUROC show "
        "that accuracy did not?\n"
        "- **Limitations:** what does this run *not* show (see Interpretation and limits)?\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/geneformer-single-cell-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/geneformer-single-cell-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Theodoris, C. V. et al. (2023). Transfer learning enables predictions in network biology. *Nature* 618, 616–624. https://www.nature.com/articles/s41586-023-06139-9\n"
        "- Chen, H. et al. (2024). Quantized multi-task learning for context-specific representations of gene network dynamics. bioRxiv 2024.08.16.608180. https://doi.org/10.1101/2024.08.16.608180\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
