"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E single-cell workflow: the pinned Geneformer V2-104M snapshot is
digest-verified (weights *and* the three gene dictionaries), cells are rank-value encoded in the
notebook, a synthetic order-sensitive dataset is validated and split, trivial baselines are
measured, a bounded AdamW fine-tuning runs in the kernel, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "geneformer-single-cell-pipeline"

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

TEMPLATE = {
    "package": "geneformer_single_cell_pipeline",
    "repo_name": REPO,
    "stem": "geneformer_single_cell",
    "notebook_name": "geneformer_single_cell_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned Geneformer V2-104M snapshot (6 files, ~421 MB: the checkpoint plus the three gene dictionaries), loads those "
        "dictionaries through a restricted unpickler, generates a deterministic 64-cell tutorial dataset over real human "
        "Ensembl gene ids (no download), validates it against the cell contract, splits it into stratified "
        "train/validation/test sets, rank-value encodes cells and inspects one encoding, computes cell embeddings, measures a "
        "majority-class and a library-size baseline on the test split, runs a bounded AdamW fine-tuning of the cell-state "
        "classification head and the last two encoder layers, evaluates accuracy, macro-F1 and AUROC on the held-out test "
        "split, classifies six freshly generated cells, exports the adapter as safetensors with a manifest, and reloads that "
        "artifact into a fresh pipeline to verify prediction parity. The default path needs no repository clone, no DIMER "
        "worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On a CPU "
        "runtime the whole path takes a few minutes, most of it in the fine-tuning cell."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled cells as a genes-as-columns CSV (`id`, one column per gene, `label`), a JSON array or a JSONL file of "
        "`{id, counts, label}` records. It passes through the same validation, stratified split, baselines, rank-value "
        "encoding, adaptation, held-out evaluation, inference, artifact export and reload-parity cells as the synthetic "
        "sample. The expected schema, the identifier requirements and the ceilings are stated in the Prerequisites and in "
        "Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
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
        "therefore cannot separate them — which is exactly the failure mode rank-value encoding exists to avoid."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and metrics modules; stage and digest-verify the "
        "Geneformer snapshot and load its gene dictionaries through a restricted unpickler; validate a labelled cell dataset "
        "against explicit ceilings and split it without leakage; read a rank-value encoding and see what it drops or truncates; "
        "extract cell embeddings; measure majority-class and library-size baselines; run a bounded fine-tuning with explicit "
        "hyperparameters and a recorded trainable-parameter set; evaluate accuracy, macro-F1 and AUROC on an independent test "
        "split; classify new cells; and export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "in-silico perturbation, gene classification, multitask or continual learning, masked-gene prediction, the V1-10M / "
        "V2-316M / cancer-tuned checkpoints, and reading `.h5ad`, `.loom` or other single-cell file formats. The repository "
        "exposes none of these; its input is a mapping of gene identifier to count."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU is enough — the default fine-tuning takes a few minutes — and CUDA is used automatically when present.",
        "- **Knowledge:** what a single-cell count matrix is, why library size differs between cells, and how accuracy, macro-F1 and AUROC differ.",
        "- **Data contract:** records are `{{id, counts, label}}` where `counts` maps a human Ensembl gene id (`ENSG...`) or an approved gene symbol to a non-negative count. At least 10 genes must be detected per cell and carry both a Geneformer token and a corpus median; at most 4,094 ranked genes fit the input; at least 8 records and 3 per class, 2..20 classes, unique ids. BYOD accepts a genes-as-columns CSV, a JSON array, or JSONL.",
        "- **Species:** the pinned vocabulary is human. Mouse or other non-human identifiers will not resolve, and the validation stage says so rather than silently dropping them.",
        "- **Privacy:** cells you bring are your responsibility. Do not upload confidential or restricted data — patient-derived or unpublished cells included — to a hosted runtime unless you are authorized to process it there. The default path uploads nothing and the BYOD branch keeps your file inside this runtime.",
        "- **Expected log lines:** loading `BertForSequenceClassification` prints that the classifier head and pooler weights are newly initialised — that is the head this tutorial trains, not a defect.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Gene dictionaries, sample cells, validation and split\n\n"
                "Geneformer's tokenizer is not a vocabulary file: it is three dictionaries shipped with the model — gene id to token, "
                "gene id to corpus median expression, and gene symbol to gene id. They are Python pickles, so this pipeline loads them "
                "through `RestrictedUnpickler`, which refuses every global except the numpy scalar types the median dictionary "
                "legitimately contains; an unexpected class raises instead of executing (NOTEBOOK_SPEC 2.0 §20). They have already been "
                "digest-verified as part of the snapshot in Section 3.\n\n"
                "The default dataset is then generated in code with a fixed seed: 32 cells per class over real human Ensembl ids, each "
                "scaled to the same library size. `validate_dataset` checks the schema, the count values, the per-cell gene coverage and "
                "the class coverage before any model runs, and — because it is given the vocabulary — also reports how many genes per "
                "cell can actually be encoded. `split_dataset` shuffles **within each class** and cuts 20 % validation / 25 % test.\n\n"
                "Look for: 20,275 tokens and 42,005 medians loaded, 64 cells, classes `['programme-A', 'programme-B']`, splits 36/12/16, "
                "and a written `outputs/{stem}_sample_dataset.csv` — the exact file shape BYOD expects. To use your own cells, set "
                "`USE_BYOD = True` and re-run from this cell."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "VAL_FRACTION = 0.2  # @param {{type:\"number\"}}\n"
                "TEST_FRACTION = 0.25  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "vocabulary = pipe.vocabulary\n"
                "print({{'tokens': len(vocabulary.tokens), 'corpus_medians': len(vocabulary.medians), 'gene_symbols': len(vocabulary.symbol_to_id), 'encodable_gene_ids': len(vocabulary.gene_ids), 'special_tokens': vocabulary.special}})\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    records = generate_sample_dataset(vocabulary)\n"
                "    data_source = f'synthetic median-matched programme dataset (seed {{SAMPLE_SEED}}, {{SAMPLE_SIZE}} cells)'\n\n"
                "dataset_manifest = validate_dataset(records, vocabulary=vocabulary)\n"
                "CLASSES = dataset_manifest['classes']\n"
                "splits = split_dataset(records, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(records, 'outputs/{stem}_sample_dataset.csv')\n\n"
                "print({{'data_source': data_source, 'n_records': dataset_manifest['n_records'], 'classes': CLASSES, 'class_counts': dataset_manifest['class_counts']}})\n"
                "print({{'detected_genes': dataset_manifest['detected_genes'], 'encodable_genes': dataset_manifest.get('encodable_genes'), 'library_size': dataset_manifest['library_size']}})\n"
                "print({{'ceilings': dataset_manifest['ceilings'], 'digest': dataset_manifest['digest'][:16] + '...'}})\n"
                "print({{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}})"
            ),
        },
        {
            "md": (
                "## 5. Read one rank-value encoding\n\n"
                "This is the cell to slow down on. `rank_value_encode` takes one cell's counts, normalises them to a fixed library size, "
                "divides each gene by its corpus median, ranks the genes by that ratio and maps them to tokens. The printout compares the "
                "top of the ranking for one cell of each class: the cells detect the same genes with near-identical total counts, yet the "
                "ranked heads differ, because the two classes put different median-matched programmes above their corpus median.\n\n"
                "The encoding also reports what it dropped — genes with a zero count, genes with no Geneformer token, genes with no corpus "
                "median — and how many ranked genes were truncated at the 4,094-token input limit. Nothing is dropped silently (VAL7)."
            ),
            "code": (
                "example_a = next(r for r in records if r['label'] == CLASSES[0])\n"
                "example_b = next(r for r in records if r['label'] == CLASSES[1])\n"
                "encoded_a = rank_value_encode(example_a['counts'], vocabulary)\n"
                "encoded_b = rank_value_encode(example_b['counts'], vocabulary)\n\n"
                "def summarise(record, encoded):\n"
                "    return {{\n"
                "        'id': record['id'],\n"
                "        'label': record['label'],\n"
                "        'library_size': encoded['library_size'],\n"
                "        'detected': encoded['n_detected'],\n"
                "        'encoded': encoded['n_encoded'],\n"
                "        'tokens_kept': encoded['n_kept'],\n"
                "        'truncated': encoded['n_truncated'],\n"
                "        'unknown_genes': len(encoded['unknown_genes']),\n"
                "        'genes_without_median': len(encoded['genes_without_median']),\n"
                "    }}\n\n"
                "print(summarise(example_a, encoded_a))\n"
                "print(summarise(example_b, encoded_b))\n"
                "print('top 8 ranked genes,', example_a['label'] + ':', encoded_a['ranked_gene_ids'][:8])\n"
                "print('top 8 ranked genes,', example_b['label'] + ':', encoded_b['ranked_gene_ids'][:8])\n"
                "print('same genes detected in both cells:', set(example_a['counts']) == set(example_b['counts']))\n"
                "print('first 8 token ids:', encoded_a['tokens'][:8])"
            ),
        },
        {
            "md": (
                "## 6. Cell embeddings (representation, not prediction)\n\n"
                "`pipe.embed` runs the verified encoder over the rank-value tokens and returns one 768-dimensional vector per cell: the "
                "mean of the last hidden state over the cell's tokens, padding excluded. Embeddings are representations — they carry no "
                "label and no metric of their own; a downstream labelled task is what gives them meaning (EVAL9). The cell embeds eight "
                "validation cells, writes them with their ids to `outputs/{stem}_embeddings.csv` (OUT4), and prints the mean cosine "
                "similarity within and between classes as an inspection, not an evaluation."
            ),
            "code": (
                "import csv\n"
                "import math\n\n"
                "embed_records = val_records[:8]\n"
                "embedding_result = pipe.embed([r['counts'] for r in embed_records], names=[r['id'] for r in embed_records])\n"
                "vectors = embedding_result['embeddings']\n"
                "print({{'n_cells': embedding_result['n_cells'], 'dimension': embedding_result['dimension'], 'tokens_kept': embedding_result['tokens_kept'][:4], 'pooling': embedding_result['pooling']}})\n\n"
                "def cosine(a, b):\n"
                "    dot = sum(x * y for x, y in zip(a, b))\n"
                "    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))\n\n"
                "within, between = [], []\n"
                "for i in range(len(embed_records)):\n"
                "    for j in range(i + 1, len(embed_records)):\n"
                "        sim = cosine(vectors[i], vectors[j])\n"
                "        (within if embed_records[i]['label'] == embed_records[j]['label'] else between).append(sim)\n"
                "print({{'mean_cosine_within_class': round(sum(within) / len(within), 4) if within else None, 'mean_cosine_between_classes': round(sum(between) / len(between), 4) if between else None, 'note': 'inspection only; embeddings are unlabelled representations'}})\n\n"
                "with open('outputs/{stem}_embeddings.csv', 'w', encoding='utf-8', newline='') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['id', 'label'] + [f'dim_{{k}}' for k in range(embedding_result['dimension'])])\n"
                "    for r, vec in zip(embed_records, vectors):\n"
                "        writer.writerow([r['id'], r['label']] + [f'{{x:.6f}}' for x in vec])\n"
                "print('wrote outputs/{stem}_embeddings.csv')"
            ),
        },
        {
            "md": (
                "## 7. Baselines on the test split\n\n"
                "Two trivial predictors set the floor before any training (EVAL10/EVAL11). `majority_baseline` predicts the most frequent "
                "training class for every test cell — 0.5 accuracy on a balanced split. `library_size_baseline` fits one threshold on total "
                "counts per cell using the training split only (SPL8); library size is the first confounder to rule out in any single-cell "
                "classification, and here it is uninformative by construction, so the baseline should land near chance. A fine-tuned model "
                "that clears both has learned something about the gene ranking rather than about sequencing depth."
            ),
            "code": (
                "baseline_majority = majority_baseline(train_records, test_records, CLASSES)\n"
                "print({{k: baseline_majority[k] for k in ('baseline', 'predicted_label', 'accuracy', 'macro_f1')}})\n"
                "if len(CLASSES) == 2:\n"
                "    baseline_library = library_size_baseline(train_records, test_records, CLASSES)\n"
                "    print({{k: baseline_library[k] for k in ('baseline', 'rule', 'train_accuracy', 'accuracy', 'macro_f1', 'auroc')}})\n"
                "else:\n"
                "    baseline_library = None\n"
                "    print('library-size baseline is defined for binary tasks only; skipped for', len(CLASSES), 'classes')"
            ),
        },
        {
            "md": (
                "## 8. Bounded fine-tuning\n\n"
                "`pipe.adapt` builds `BertForSequenceClassification` from the verified checkpoint (the head and pooler are newly "
                "initialised — the log line says so), freezes every parameter except the head, the pooler and the last `TRAINABLE_LAYERS` "
                "encoder layers, and runs AdamW with the hyperparameters below (FT4/FT6): tutorial values chosen for a few minutes of CPU, "
                "not production settings. Validation metrics are computed after each epoch for **monitoring only**; the final epoch's "
                "weights are kept, so no selection happens on the validation split (EVAL14). Training loss going down is optimisation "
                "evidence, not task-quality evidence (FT7) — Section 9 is where quality is measured.\n\n"
                "Look for about 14.8 M trainable parameters of 104 M, and validation accuracy leaving 0.5 partway through training. If it "
                "is still at 0.5 in the last epoch the head is under-trained rather than broken: the AUROC in the next section will be high "
                "while accuracy sits at chance, which means the ranking is right and the decision boundary has not moved yet."
            ),
            "code": (
                "import time\n\n"
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 2e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_LAYERS = 2  # @param {{type:\"integer\"}}\n\n"
                "started = time.perf_counter()\n"
                "adapt_result = pipe.adapt(\n"
                "    train_records,\n"
                "    val_records,\n"
                "    classes=CLASSES,\n"
                "    epochs=EPOCHS,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    trainable_layers=TRAINABLE_LAYERS,\n"
                "    seed=SEED,\n"
                ")\n"
                "adapt_seconds = round(time.perf_counter() - started, 1)\n"
                "print({{'method': adapt_result['method'], 'trainable_parameters': adapt_result['trainable_parameters'], 'total_parameters': adapt_result['total_parameters'], 'precision': adapt_result['precision'], 'device': pipe.device, 'seconds': adapt_seconds}})\n"
                "for step in adapt_result['history']:\n"
                "    print(step)"
            ),
        },
        {
            "md": (
                "## 9. Held-out evaluation\n\n"
                "`pipe.evaluate` classifies every cell of a split and reports `accuracy` (discrete correctness under the argmax rule), "
                "`macro_f1` (the unweighted mean of per-class F1, which exposes a model that ignores a class), per-class precision/recall/F1 "
                "with support, and `auroc` (ranking quality of the positive-class score, independent of the argmax threshold). The **test "
                "split** was never used for training or monitoring, so its numbers are the independent evidence (SPL6/SPL7). These are "
                "tutorial metrics on a synthetic 16-cell split (EVAL6): one holdout, no dispersion estimate. The report, with both baselines "
                "and the deltas against them, is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "val_metrics = pipe.evaluate(val_records)\n"
                "test_metrics = pipe.evaluate(test_records)\n"
                "print({{'split': 'validation', **{{k: val_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "print({{'split': 'test', **{{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "for cls_name, row in test_metrics['per_class'].items():\n"
                "    print({{'class': cls_name, **row}})\n\n"
                "evaluation_report = {{\n"
                "    'task': 'single-cell state classification (bounded fine-tuning of Geneformer V2-104M)',\n"
                "    'evidence': 'tutorial sample-sanity metrics on one stratified holdout; not a benchmark and not biology',\n"
                "    'estimation': 'single train/validation/test split, seed ' + str(SEED) + ', no dispersion estimate',\n"
                "    'data_source': data_source,\n"
                "    'dataset_digest': dataset_manifest['digest'],\n"
                "    'classes': CLASSES,\n"
                "    'splits': {{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}},\n"
                "    'baselines': {{'majority': baseline_majority, 'library_size': baseline_library}},\n"
                "    'validation_metrics': val_metrics,\n"
                "    'test_metrics': test_metrics,\n"
                "    'delta_vs_majority': {{k: round(test_metrics[k] - baseline_majority[k], 4) for k in ('accuracy', 'macro_f1')}},\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k != 'trainable_parameter_names'}},\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "print({{'delta_vs_majority': evaluation_report['delta_vs_majority'], 'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 10. Inference on new cells\n\n"
                "`pipe.classify` returns, per cell, the argmax `label`, its `score` and the full `scores` dictionary in class order. The "
                "scores are softmax outputs of a head trained on a few dozen cells — **not calibrated probabilities** (UNC2); the only "
                "decision rule is argmax (UNC3), and a deployment that must trade false positives against false negatives owns its own "
                "threshold. On the default path the new cells are generated with a different seed, so their true labels are known and shown "
                "as a check; on the BYOD path the first six test-split cells stand in as new data (INF2). Predictions are written to "
                "`outputs/{stem}_predictions.csv` with ids and per-class scores."
            ),
            "code": (
                "if USE_BYOD:\n"
                "    new_records = test_records[:6]\n"
                "    new_source = 'first six BYOD test-split cells'\n"
                "else:\n"
                "    new_records = generate_sample_dataset(vocabulary, seed=7, size=6)\n"
                "    new_source = 'freshly generated cells (seed 7)'\n"
                "input_manifest = validate_inputs([r['counts'] for r in new_records], vocabulary, names=[r['id'] for r in new_records])\n"
                "print({{'new_source': new_source, 'verdict': input_manifest['verdict'], 'n_cells': input_manifest['n_cells'], 'max_tokens_observed': input_manifest['max_tokens_observed']}})\n"
                "inference_result = pipe.classify([r['counts'] for r in new_records], names=[r['id'] for r in new_records])\n"
                "predictions = inference_result['predictions']\n"
                "print({{'decision_rule': inference_result['decision_rule']}})\n"
                "n_match = 0\n"
                "for p, r in zip(predictions, new_records):\n"
                "    n_match += p['label'] == r['label']\n"
                "    print({{'id': p['id'], 'predicted': p['label'], 'score': round(p['score'], 4), 'true_label': r['label']}})\n"
                "print({{'matches': n_match, 'of': len(new_records), 'note': 'sanity check on generated labels, not an evaluation'}})\n\n"
                "with open('outputs/{stem}_predictions.csv', 'w', encoding='utf-8', newline='') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['id', 'tokens_kept', 'predicted_label', 'score'] + [f'score_{{c}}' for c in CLASSES])\n"
                "    for p in predictions:\n"
                "        writer.writerow([p['id'], p['tokens_kept'], p['label'], f\"{{p['score']:.6f}}\"] + [f\"{{p['scores'][c]:.6f}}\" for c in CLASSES])\n"
                "print('wrote outputs/{stem}_predictions.csv')"
            ),
        },
        {
            "md": (
                "## 11. Export the adapter and verify a fresh reload\n\n"
                "`pipe.save_artifact` writes only the trained tensors (head, pooler and the unfrozen encoder layers) as "
                "`adapter.safetensors` plus a `manifest.json` that records the artifact format, the exact base model id, revision **and "
                "subfolder** the tensors belong to (ART4 — the repository hosts several checkpoints, so the subfolder is part of the "
                "identity), the class order, the tensor names, the file size and SHA-256, and the adaptation configuration (OUT8). "
                "`GeneformerPipeline.from_artifact` then re-verifies the base snapshot, checks the artifact manifest and digests **before** "
                "deserialising, rebuilds the classifier and overlays the tensors — a fresh object from files, not the in-memory model "
                "(VER2). The cell compares its predictions on the same new cells with the pre-export ones: labels must match exactly and "
                "scores within `1e-5` (VER4)."
            ),
            "code": (
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "pipe.save_artifact(artifact_dir, metadata={{'data_source': data_source, 'dataset_digest': dataset_manifest['digest'], 'test_metrics': {{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "with open(artifact_dir / ARTIFACT_MANIFEST_NAME, encoding='utf-8') as f:\n"
                "    artifact_manifest = json.load(f)\n"
                "print({{'format': artifact_manifest['format'], 'base_model': artifact_manifest['base_model'], 'classes': artifact_manifest['classes'], 'n_tensors': len(artifact_manifest['tensors']), 'files': artifact_manifest['files']}})\n\n"
                "reloaded_pipe = GeneformerPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR)\n"
                "reloaded_result = reloaded_pipe.classify([r['counts'] for r in new_records], names=[r['id'] for r in new_records])\n"
                "max_score_diff = 0.0\n"
                "for before, after in zip(predictions, reloaded_result['predictions']):\n"
                "    assert before['id'] == after['id'] and before['label'] == after['label'], f'reload parity failure on {{before[\"id\"]}}'\n"
                "    max_score_diff = max(max_score_diff, abs(before['score'] - after['score']))\n"
                "assert max_score_diff < 1e-5, f'reload score drift {{max_score_diff}}'\n"
                "print({{'reload_parity': 'PASS', 'labels_equal': True, 'max_abs_score_diff': max_score_diff, 'loaded_from': reloaded_pipe.adaptation.get('loaded_from_artifact')}})"
            ),
        },
        {
            "md": (
                "## 12. Result export and provenance\n\n"
                "The last output, `outputs/{stem}_result.json`, gathers everything a reader needs to interpret the files above: the "
                "notebook source revision, the model id, immutable revision, subfolder and licence, the dataset source and digest, the "
                "adaptation configuration, baseline and held-out metrics, the new-cell predictions, the artifact manifest, the "
                "reload-parity result, and the runtime versions and device (OUT6/OUT7). No credential is involved anywhere in this "
                "notebook, so none can leak into it (OUT10)."
            ),
            "code": (
                "import platform\n\n"
                "result_payload = {{\n"
                "    'task': 'single-cell state classification adaptation (Geneformer V2-104M)',\n"
                "    'pipeline_class': 'GeneformerPipeline',\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_subfolder': MODEL_SUBFOLDER,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'data_source': data_source,\n"
                "    'dataset_manifest': dataset_manifest,\n"
                "    'vocabulary': {{'tokens': len(vocabulary.tokens), 'corpus_medians': len(vocabulary.medians), 'special_tokens': vocabulary.special}},\n"
                "    'encoding_example': {{'id': example_a['id'], 'label': example_a['label'], 'top_ranked_gene_ids': encoded_a['ranked_gene_ids'][:8], 'tokens_kept': encoded_a['n_kept'], 'truncated': encoded_a['n_truncated']}},\n"
                "    'embedding_summary': {{'n_cells': embedding_result['n_cells'], 'dimension': embedding_result['dimension'], 'pooling': embedding_result['pooling']}},\n"
                "    'evaluation_report': evaluation_report,\n"
                "    'inference': {{'new_source': new_source, 'decision_rule': inference_result['decision_rule'], 'predictions': predictions}},\n"
                "    'artifact_format': ARTIFACT_FORMAT,\n"
                "    'artifact_format_version': ARTIFACT_FORMAT_VERSION,\n"
                "    'artifact_manifest': artifact_manifest,\n"
                "    'reload_parity': {{'labels_equal': True, 'max_abs_score_diff': max_score_diff}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'safetensors': safetensors.__version__,\n"
                "        'device': pipe.device,\n"
                "        'precision': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
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
        "**Optional experiments (do not affect the default path):** set `TRAINABLE_LAYERS = 0` to train the head alone and compare the test "
        "metrics; lower `EPOCHS` to 2 to see the under-trained regime where AUROC is high but accuracy sits at 0.5 — ranking before "
        "thresholding; or bring a real labelled dataset through BYOD and read the library-size baseline first, because if it already scores "
        "well your labels may be predictable from sequencing depth alone.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/geneformer-single-cell-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/geneformer-single-cell-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Theodoris, C. V. et al. (2023). Transfer learning enables predictions in network biology. *Nature* 618, 616–624. https://www.nature.com/articles/s41586-023-06139-9\n"
        "- Chen, H. et al. (2024). Quantized multi-task learning for context-specific representations of gene network dynamics. bioRxiv 2024.08.16.608180. https://doi.org/10.1101/2024.08.16.608180"
    ),
}
