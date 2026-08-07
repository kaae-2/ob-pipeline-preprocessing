# Preprocessing Repository

## What this module does

This module converts imported dataset tarballs into benchmark-ready train/test
splits.

- Main entrypoint: `data_preprocessing.py`
- Local runner: `run_data_preprocessing.sh`
- Key outputs:
  - `*.train.matrix.tar.gz`
  - `*.train.labels.tar.gz`
  - `*.test.matrices.tar.gz`
  - `*.test.labels.tar.gz`
  - `*.metadata.json.gz`
  - `*.split_audit.json.gz`

Label handling maps canonical non-target labels (`""`, `unlabeled`, `ungated`,
`debris`, `unknown`, `other`, `noise`) and numeric zero to label id `0`.

## Split-audit schema

`metadata.split_audit` is the canonical split provenance consumed downstream.
The dedicated `*.split_audit.json.gz` output is an exact deterministic JSON
mirror. Schema version `1.0.0` contains:

- `identities`: the complete incoming dataset metadata and data-import
  parameters, plus all effective preprocessing CLI parameters.
- `split`: requested/effective fold, wrapping status and reason, selected
  training sample, and ordered test samples.
- `counts.training.nominal_rows`: rows in the selected raw training sample
  before sub-sampling or label eligibility.
- `counts.training.sampled_rows`: rows written after configured deterministic
  sub-sampling, including label `0`.
- `counts.training.eligible_rows`: positive-label rows after sub-sampling.
- `counts.test`: total and per-sample rows before/after filtering. Both values
  are equal at preprocessing because filtering belongs to stratify.
- `populations`: one record for each positive metadata label id. The nominal
  count is before sub-sampling; `training_support` is the positive-label count
  written for model training; `test_truth_count` is aggregated over ordered
  test samples; `present_in_training` is exactly `training_support > 0`.
- `non_target`: label id `0` counts. It represents ungated, invalid, or
  rejection labels and is explicitly not a biological population.

Stratify must recompute final support and filtering totals from its output
archives. Preprocessing counts are base-split provenance, not final evidence
when a downstream filter changes row eligibility.

## Run locally

```bash
bash preprocessing/run_data_preprocessing.sh
```

Or direct CLI:

```bash
python preprocessing/data_preprocessing.py --name data_import --output_dir preprocessing/out/... --data.raw <dataset>.data.tar.gz --data.metadata <dataset>.metadata.json.gz --num 1
```

## Run as part of benchmark

Configured in `benchmark/Clustering_conda.yml` preprocessing stage. Run through:

```bash
just benchmark
```

## What `run_data_preprocessing.sh` needs

- Existing data-import outputs in `preprocessing/out/data/data_import/`
- Python environment with `pandas` and `numpy` (`pyarrow` optional)
- Writable output directories (script writes into `preprocessing/out/...`)
