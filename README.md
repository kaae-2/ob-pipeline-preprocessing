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

Label handling maps canonical non-target labels (`""`, `unlabeled`, `ungated`,
`debris`, `unknown`, `other`, `noise`) to label id `0`.

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
