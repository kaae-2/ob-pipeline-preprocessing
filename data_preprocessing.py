#!/usr/bin/env python

"""Preprocess imported CSV tarballs into benchmark train/test archives."""

import argparse
import csv
import gzip
import json
import os
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False


LABEL_COLUMN_CANDIDATES = (
    'label',
    'population',
    'cell_type',
    'celltype',
    'cluster',
    'cluster_id',
)
UNLABELED_VALUES = {'', 'unlabeled', 'ungated', 'debris', 'unknown', 'other', 'noise'}
TAR_GZIP_COMPRESSLEVEL = 1


def read_bytes_handling_gzip(path: str) -> bytes:
    try:
        with gzip.open(path, 'rb') as fh:
            return fh.read()
    except (OSError, gzip.BadGzipFile):
        with open(path, 'rb') as fh:
            return fh.read()


def is_tar_archive(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return tarfile.is_tarfile(path)
    except (OSError, tarfile.TarError):
        return False


def extract_csv_from_tar(tar_path: Path) -> Tuple[tempfile.TemporaryDirectory, List[Path]]:
    tmp_dir = tempfile.TemporaryDirectory()
    with tarfile.open(tar_path, mode='r:*') as tar:
        members = [
            m
            for m in tar.getmembers()
            if m.name.lower().endswith('.csv') or m.name.lower().endswith('.csv.gz')
        ]
        if not members:
            tmp_dir.cleanup()
            raise FileNotFoundError(f'No CSV files found in archive: {tar_path}')
        extracted: List[Path] = []
        for member in members:
            tar.extract(member, path=tmp_dir.name, filter='data')
            extracted.append(Path(tmp_dir.name) / member.name)
    return tmp_dir, sorted(extracted, key=lambda p: p.name)


def read_csv_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.gz' or path.name.lower().endswith('.csv.gz'):
        if _PYARROW_AVAILABLE:
            return pd.read_csv(path, compression='gzip', engine='pyarrow')
        return pd.read_csv(path, compression='gzip')
    if _PYARROW_AVAILABLE:
        return pd.read_csv(path, engine='pyarrow')
    return pd.read_csv(path)


def read_csv_header(path: Path) -> List[str]:
    if path.suffix.lower() == '.gz' or path.name.lower().endswith('.csv.gz'):
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as fh:
            reader = csv.reader(fh)
            try:
                return next(reader)
            except StopIteration as exc:
                raise ValueError(f'CSV file has no header row: {path.name}') from exc

    with open(path, 'r', encoding='utf-8', newline='') as fh:
        reader = csv.reader(fh)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f'CSV file has no header row: {path.name}') from exc


def _read_csv_column(path: Path, column_name: str) -> pd.Series:
    if path.suffix.lower() == '.gz' or path.name.lower().endswith('.csv.gz'):
        if _PYARROW_AVAILABLE:
            df = pd.read_csv(
                path, compression='gzip', usecols=[column_name], engine='pyarrow'
            )
        else:
            df = pd.read_csv(path, compression='gzip', usecols=[column_name])
    else:
        if _PYARROW_AVAILABLE:
            df = pd.read_csv(path, usecols=[column_name], engine='pyarrow')
        else:
            df = pd.read_csv(path, usecols=[column_name])
    return df[column_name]


def _read_csv_row_count(path: Path, first_column: str) -> int:
    if path.suffix.lower() == '.gz' or path.name.lower().endswith('.csv.gz'):
        if _PYARROW_AVAILABLE:
            df = pd.read_csv(
                path, compression='gzip', usecols=[first_column], engine='pyarrow'
            )
        else:
            df = pd.read_csv(path, compression='gzip', usecols=[first_column])
    else:
        if _PYARROW_AVAILABLE:
            df = pd.read_csv(path, usecols=[first_column], engine='pyarrow')
        else:
            df = pd.read_csv(path, usecols=[first_column])
    return len(df)


def find_label_column_from_headers(headers: Sequence[object]) -> Optional[str]:
    lower_map = {str(col).strip().lower(): str(col) for col in headers}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def extract_labels_with_column(
    df: pd.DataFrame, label_col: Optional[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    if label_col and label_col in df.columns:
        labels = df[label_col]
        features = df.drop(columns=[label_col])
        return features, labels

    detected = find_label_column_from_headers(df.columns)
    if detected is None:
        labels = pd.Series(['unlabeled'] * len(df), name='label')
        return df, labels
    labels = df[detected]
    features = df.drop(columns=[detected])
    return features, labels


def _load_csv_sample(
    sample_id: int, csv_path: Path, label_col: Optional[str]
) -> Tuple[int, pd.DataFrame, pd.Series]:
    df = read_csv_dataframe(csv_path)
    features, labels = extract_labels_with_column(df, label_col)
    return sample_id, features, labels


def _collect_sample_label_values(
    sample_id: int, csv_path: Path, preferred_label_col: Optional[str]
) -> Tuple[int, set[str]]:
    header = read_csv_header(csv_path)
    if preferred_label_col and preferred_label_col in header:
        label_col = preferred_label_col
    else:
        label_col = find_label_column_from_headers(header)

    if label_col is None:
        if not header:
            raise ValueError(f'CSV file has no header row: {csv_path.name}')
        _read_csv_row_count(csv_path, header[0])
        return sample_id, set()

    labels = _read_csv_column(csv_path, label_col)
    normalized = _normalize_label_series(labels)
    values = normalized.dropna().astype(str).str.strip()
    return sample_id, {
        value for value in values if value and value.lower() not in UNLABELED_VALUES
    }


def build_label_key_from_values(label_values: Sequence[str]) -> Dict[int, str]:
    ordered = sorted(
        {
            value
            for value in label_values
            if value and value.lower() not in UNLABELED_VALUES
        }
    )
    return {idx + 1: label for idx, label in enumerate(ordered)}


def map_labels_to_ints(labels: pd.Series, id_to_label: Dict[int, str]) -> pd.Series:
    label_to_id = {label: idx for idx, label in id_to_label.items()}
    mapped: List[int] = []
    for value in labels:
        if pd.isna(value):
            mapped.append(0)
            continue
        text = str(value).strip()
        if not text or text.lower() in UNLABELED_VALUES:
            mapped.append(0)
            continue
        mapped.append(label_to_id.get(text, 0))
    return pd.Series(mapped, name='label')


def write_csv(df: pd.DataFrame, path: Path, header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, header=header)


def write_series_csv(series: pd.Series, path: Path, header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, index=False, header=header)


def _write_label_key(
    out_dir: Path,
    name: str,
    id_to_label: Dict[int, str],
    dataset_name: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {'id_to_label': {str(k): v for k, v in id_to_label.items()}}
    metadata: Dict[str, str] = {}
    if dataset_name is not None and dataset_name.strip():
        metadata['dataset_name'] = dataset_name.strip()
    if metadata:
        payload['metadata'] = metadata
    key_path = out_dir / f'{name}.label_key.json.gz'
    with gzip.open(key_path, 'wt') as handle:
        json.dump(payload, handle, indent=2)


def _write_sample_archives_from_paths(
    out_dir: Path,
    name: str,
    train_id: int,
    test_ids: Sequence[int],
    sample_paths: Dict[int, Path],
    id_to_label: Dict[int, str],
    preferred_label_col: Optional[str],
    sub_sampling: int,
) -> None:
    train_matrix_path = out_dir / f'{name}.train.matrix.tar.gz'
    train_labels_path = out_dir / f'{name}.train.labels.tar.gz'
    test_matrices_path = out_dir / f'{name}.test.matrices.tar.gz'
    test_labels_path = out_dir / f'{name}.test.labels.tar.gz'

    with tempfile.TemporaryDirectory() as tmpdir:
        _, train_features, train_labels = _load_csv_sample(
            train_id, sample_paths[train_id], preferred_label_col
        )
        train_labels = _normalize_label_series(train_labels)
        if sub_sampling > 0:
            train_features, train_labels = _subsample_training_data(
                train_features, train_labels, sub_sampling
            )
        mapped_train_labels = map_labels_to_ints(train_labels, id_to_label)

        train_matrix_file = Path(tmpdir) / f'{name}-data-{train_id}.csv'
        train_label_file = Path(tmpdir) / f'{name}-label-{train_id}.csv'
        write_csv(train_features, train_matrix_file)
        write_series_csv(mapped_train_labels, train_label_file)

        with tarfile.open(
            train_matrix_path, 'w:gz', compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            tar.add(train_matrix_file, arcname=train_matrix_file.name)

        with tarfile.open(
            train_labels_path, 'w:gz', compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            tar.add(train_label_file, arcname=train_label_file.name)

        test_matrix_files: List[Path] = []
        test_label_files: List[Path] = []
        for test_id in test_ids:
            _, test_features, test_labels = _load_csv_sample(
                test_id, sample_paths[test_id], preferred_label_col
            )
            test_labels = _normalize_label_series(test_labels)
            mapped_test_labels = map_labels_to_ints(test_labels, id_to_label)

            matrix_file = Path(tmpdir) / f'{name}-data-{test_id}.csv'
            label_file = Path(tmpdir) / f'{name}-label-{test_id}.csv'
            write_csv(test_features, matrix_file)
            write_series_csv(mapped_test_labels, label_file)
            test_matrix_files.append(matrix_file)
            test_label_files.append(label_file)

        with tarfile.open(
            test_matrices_path, 'w:gz', compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            for path in test_matrix_files:
                tar.add(path, arcname=path.name)

        with tarfile.open(
            test_labels_path, 'w:gz', compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            for path in test_label_files:
                tar.add(path, arcname=path.name)


def load_order_payload(order_path: str) -> Dict[str, object]:
    try:
        payload = json.loads(read_bytes_handling_gzip(order_path).decode('utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Failed to parse order JSON: {order_path}') from exc
    if not isinstance(payload, dict) or 'order' not in payload:
        raise ValueError("Order JSON must be an object with an 'order' key.")
    order = payload.get('order')
    if not isinstance(order, list) or not order:
        raise ValueError('Order JSON must contain a non-empty list.')
    if not all(isinstance(item, int) for item in order):
        raise ValueError('Order entries must be integers.')
    if len(set(order)) != len(order):
        raise ValueError('Order entries must be unique.')
    return payload


def _normalize_label_series(labels: pd.Series) -> pd.Series:
    normalized = labels.copy()
    stripped = normalized.astype(str).str.strip()
    lower = stripped.str.lower()
    mask = normalized.isna() | lower.isin(UNLABELED_VALUES)
    normalized = stripped
    normalized.loc[mask] = 'unlabeled'
    return normalized


def _subsample_training_data(
    features: pd.DataFrame, labels: pd.Series, target_size: int
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_size <= 0:
        return features, labels
    total = len(labels)
    if target_size >= total:
        print(
            'Warning: sub-sampling size exceeds training set; using all cells.',
            file=sys.stderr,
        )
        return features, labels

    counts = labels.value_counts(dropna=False)
    expected = counts / total * target_size
    base = expected.apply(np.floor).astype(int)
    base = base.clip(upper=counts)
    allocated = int(base.sum())
    remaining = target_size - allocated

    if remaining > 0:
        fractional = (expected - base).rename('fractional')
        tie_break = counts.rename('count')
        priority = (
            pd.concat([fractional, tie_break], axis=1)
            .sort_values(by=['fractional', 'count'], ascending=[False, True])
            .index
        )
        for label in priority:
            if remaining <= 0:
                break
            if base[label] < counts[label]:
                base[label] += 1
                remaining -= 1

    rng = np.random.default_rng(0)
    selected_indices: List[int] = []
    for label, take_count in base.items():
        if take_count <= 0:
            continue
        label_indices = labels[labels == label].index.to_numpy()
        if take_count >= len(label_indices):
            chosen = label_indices
        else:
            chosen = rng.choice(label_indices, size=take_count, replace=False)
        selected_indices.extend(chosen.tolist())

    selected_indices = sorted(selected_indices)
    return features.loc[selected_indices], labels.loc[selected_indices]


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Preprocess imported CSV tarballs into train/test archives.'
    )
    parser.add_argument(
        '--data.raw',
        type=str,
        required=True,
        help='Path to imported tar.gz archive containing CSV/CSV.GZ samples.',
    )
    parser.add_argument(
        '--data.order',
        type=str,
        required=True,
        help="Path to JSON file containing an 'order' list of 1-based sample indices.",
    )
    parser.add_argument(
        '--num',
        type=int,
        required=True,
        help='1-based index into data.order for the training sample.',
    )
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--name', type=str, default='dataset')
    parser.add_argument('--test-sample-limit', type=int, default=None)
    parser.add_argument('--max-workers', type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = parse_args()
    args = parser.parse_args(argv)

    raw_path = Path(getattr(args, 'data.raw'))
    order_path = getattr(args, 'data.order')
    output_dir = args.output_dir
    name = args.name
    num = args.num
    test_sample_limit = args.test_sample_limit
    max_workers_arg = args.max_workers

    if not is_tar_archive(raw_path):
        raise ValueError(
            '--data.raw must be a tar/tar.gz archive produced by data_import.'
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    order_payload = load_order_payload(order_path)
    order = order_payload['order']
    metadata = order_payload.get('metadata')
    if not isinstance(metadata, dict):
        metadata = {}

    dataset_name_raw = metadata.get('dataset_name')
    dataset_name: Optional[str] = None
    if isinstance(dataset_name_raw, str) and dataset_name_raw.strip():
        dataset_name = dataset_name_raw.strip()

    sub_sampling_raw = metadata.get('sub_sampling', 0)
    try:
        if isinstance(sub_sampling_raw, bool):
            sub_sampling = int(sub_sampling_raw)
        elif isinstance(sub_sampling_raw, (int, float)):
            sub_sampling = int(sub_sampling_raw)
        elif isinstance(sub_sampling_raw, str) and sub_sampling_raw.strip().isdigit():
            sub_sampling = int(sub_sampling_raw.strip())
        else:
            raise ValueError
    except Exception:
        print('Warning: invalid sub_sampling metadata; defaulting to 0.', file=sys.stderr)
        sub_sampling = 0
    if sub_sampling < 0:
        print('Warning: sub_sampling metadata must be non-negative; defaulting to 0.', file=sys.stderr)
        sub_sampling = 0

    tmp_dir, csv_paths = extract_csv_from_tar(raw_path)
    try:
        if not csv_paths:
            raise FileNotFoundError(f'No CSV inputs found at {raw_path}')

        sample_count = len(csv_paths)
        if set(order) != set(range(1, sample_count + 1)):
            raise ValueError('Order must contain each sample index exactly once (1..n).')
        if len(order) != sample_count:
            raise ValueError('Order length must match number of samples.')
        if num < 1:
            raise ValueError('num must be within 1..n.')
        if num > sample_count:
            wrapped_num = ((num - 1) % sample_count) + 1
            print(
                'Warning: num {} exceeds sample_count {}; using num {}. '
                'Duplicate folds will be filtered in metrics.'.format(
                    num, sample_count, wrapped_num
                ),
                file=sys.stderr,
            )
            num = wrapped_num

        train_pos = num - 1
        train_id = order[train_pos]

        max_test = max(sample_count - 1, 0)
        if test_sample_limit is None:
            print('# --test-sample-limit not provided; using all remaining samples')
            test_count = max_test
        else:
            if test_sample_limit <= 0:
                raise ValueError('test-sample-limit must be a positive integer.')
            if test_sample_limit > max_test:
                print(
                    'Warning: test-sample-limit exceeds n-1; using all remaining samples.',
                    file=sys.stderr,
                )
            test_count = min(test_sample_limit, max_test)

        test_ids: List[int] = []
        for offset in range(1, test_count + 1):
            test_ids.append(order[(train_pos + offset) % sample_count])

        first_header = read_csv_header(csv_paths[0])
        label_col = find_label_column_from_headers(first_header)

        indexed_paths = list(enumerate(csv_paths, start=1))
        sample_paths = {idx: path for idx, path in indexed_paths}
        max_workers = min(64, max(4, (os.cpu_count() or 1) * 2), len(indexed_paths))
        if max_workers_arg is not None:
            if max_workers_arg <= 0:
                raise ValueError('max-workers must be a positive integer.')
            max_workers = min(max_workers_arg, len(indexed_paths))

        label_values: set[str] = set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_collect_sample_label_values, idx, path, label_col)
                for idx, path in indexed_paths
            ]
            for future in as_completed(futures):
                _, sample_values = future.result()
                label_values.update(sample_values)

        id_to_label = build_label_key_from_values(sorted(label_values))
        _write_sample_archives_from_paths(
            out_dir=out_dir,
            name=name,
            train_id=train_id,
            test_ids=test_ids,
            sample_paths=sample_paths,
            id_to_label=id_to_label,
            preferred_label_col=label_col,
            sub_sampling=sub_sampling,
        )
        _write_label_key(out_dir, name, id_to_label, dataset_name=dataset_name)
    finally:
        tmp_dir.cleanup()


if __name__ == '__main__':
    main()
