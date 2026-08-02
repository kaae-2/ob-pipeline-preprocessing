#!/usr/bin/env python

"""Preprocess imported CSV tarballs into benchmark train/test archives."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import uuid
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
PREPROCESS_CACHE_ROOT = Path(__file__).resolve().parents[1] / '.cache' / 'preprocessing'


def _copy_json_object(value):
    return json.loads(json.dumps(value))


def _is_non_target_label(value: object) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    if not text or text.lower() in UNLABELED_VALUES:
        return True
    try:
        return float(text) == 0.0
    except ValueError:
        return False


def _populate_metadata_legacy_aliases(payload: Dict[str, object]) -> Dict[str, object]:
    dataset = payload.get('dataset')
    samples = payload.get('samples')
    labels = payload.get('labels')
    stages = payload.get('stages')

    legacy_metadata: Dict[str, object] = {}
    if isinstance(dataset, dict):
        legacy_metadata.update(_copy_json_object(dataset))
    if isinstance(samples, dict):
        for key in ('sample_names', 'cells_per_sample', 'sample_count'):
            if key in samples:
                legacy_metadata[key] = _copy_json_object(samples[key])
    if isinstance(stages, dict):
        stratify = stages.get('stratify')
        if isinstance(stratify, dict) and 'stratification' in stratify:
            legacy_metadata['stratification'] = _copy_json_object(
                stratify['stratification']
            )

    payload['metadata'] = legacy_metadata
    if isinstance(samples, dict) and 'order' in samples:
        payload['order'] = _copy_json_object(samples['order'])
    if isinstance(labels, dict):
        if 'id_to_label' in labels:
            payload['id_to_label'] = _copy_json_object(labels['id_to_label'])
        if 'label_to_id' in labels:
            payload['label_to_id'] = _copy_json_object(labels['label_to_id'])
    return payload


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
        member_names = [member.name for member in members]
        if len(set(member_names)) != len(member_names):
            tmp_dir.cleanup()
            raise ValueError(f'Duplicate CSV member names found in archive: {tar_path}')
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


def _archive_cache_dir(raw_path: Path) -> Path:
    stat = raw_path.stat()
    key_material = f'{raw_path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}'
    digest = hashlib.sha256(key_material.encode('utf-8')).hexdigest()
    return PREPROCESS_CACHE_ROOT / digest


def _sample_cache_paths(
    cache_dir: Path, sample_id: int, csv_path: Path, label_col: Optional[str]
) -> tuple[Path, Path]:
    label_key = (label_col or 'auto').strip().lower() or 'auto'
    sample_key = hashlib.sha256(
        f'{sample_id}:{csv_path.name}:{label_key}'.encode('utf-8')
    ).hexdigest()[:16]
    return (
        cache_dir / f'{sample_key}.features.pkl.gz',
        cache_dir / f'{sample_key}.labels.pkl.gz',
    )


def _load_cached_csv_sample(
    sample_id: int,
    csv_path: Path,
    label_col: Optional[str],
    cache_dir: Path,
) -> Tuple[int, pd.DataFrame, pd.Series]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_cache, labels_cache = _sample_cache_paths(
        cache_dir, sample_id, csv_path, label_col
    )
    if features_cache.exists() and labels_cache.exists():
        features = pd.read_pickle(features_cache, compression='gzip')
        labels = pd.read_pickle(labels_cache, compression='gzip')
        return sample_id, features, labels

    _, features, labels = _load_csv_sample(sample_id, csv_path, label_col)
    tmp_suffix = f'.tmp.{os.getpid()}.{uuid.uuid4().hex}'
    tmp_features = features_cache.with_name(f'{features_cache.name}{tmp_suffix}')
    tmp_labels = labels_cache.with_name(f'{labels_cache.name}{tmp_suffix}')
    try:
        features.to_pickle(tmp_features, compression='gzip')
        labels.to_pickle(tmp_labels, compression='gzip')
        tmp_features.replace(features_cache)
        tmp_labels.replace(labels_cache)
    finally:
        tmp_features.unlink(missing_ok=True)
        tmp_labels.unlink(missing_ok=True)
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
    return sample_id, {value for value in values if not _is_non_target_label(value)}


def build_label_key_from_values(label_values: Sequence[str]) -> Dict[int, str]:
    ordered = sorted(
        {
            value
            for value in label_values
            if not _is_non_target_label(value)
        }
    )
    return {idx + 1: label for idx, label in enumerate(ordered)}


def map_labels_to_ints(labels: pd.Series, id_to_label: Dict[int, str]) -> pd.Series:
    label_to_id = {label: idx for idx, label in id_to_label.items()}
    mapped: List[int] = []
    for value in labels:
        text = str(value).strip()
        if _is_non_target_label(value):
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


def _write_json_gzip(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as raw_handle, gzip.GzipFile(
        filename='', fileobj=raw_handle, mode='wb', mtime=0
    ) as gzip_handle:
        serialized = json.dumps(payload, indent=2, sort_keys=True) + '\n'
        gzip_handle.write(serialized.encode('utf-8'))


def _write_metadata(out_dir: Path, name: str, payload: Dict[str, object]) -> None:
    _write_json_gzip(
        out_dir / f'{name}.metadata.json.gz',
        _populate_metadata_legacy_aliases(payload),
    )


def _label_counts(labels: pd.Series) -> Dict[int, int]:
    return {
        int(label_id): int(count)
        for label_id, count in labels.value_counts().sort_index().items()
    }


def _write_sample_archives_from_paths(
    out_dir: Path,
    name: str,
    train_id: int,
    test_ids: Sequence[int],
    sample_paths: Dict[int, Path],
    id_to_label: Dict[int, str],
    preferred_label_col: Optional[str],
    sub_sampling: int,
    cache_dir: Path,
) -> Dict[str, object]:
    train_matrix_path = out_dir / f'{name}.train.matrix.tar.gz'
    train_labels_path = out_dir / f'{name}.train.labels.tar.gz'
    test_matrices_path = out_dir / f'{name}.test.matrices.tar.gz'
    test_labels_path = out_dir / f'{name}.test.labels.tar.gz'

    with tempfile.TemporaryDirectory() as tmpdir:
        _, train_features, train_labels = _load_cached_csv_sample(
            train_id, sample_paths[train_id], preferred_label_col, cache_dir
        )
        train_labels = _normalize_label_series(train_labels)
        nominal_train_labels = map_labels_to_ints(train_labels, id_to_label)
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
        mapped_test_by_id: Dict[int, pd.Series] = {}
        for test_id in test_ids:
            _, test_features, test_labels = _load_cached_csv_sample(
                test_id, sample_paths[test_id], preferred_label_col, cache_dir
            )
            test_labels = _normalize_label_series(test_labels)
            mapped_test_labels = map_labels_to_ints(test_labels, id_to_label)
            mapped_test_by_id[test_id] = mapped_test_labels

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

    nominal_train_counts = _label_counts(nominal_train_labels)
    sampled_train_counts = _label_counts(mapped_train_labels)
    test_counts: Dict[int, int] = {}
    for labels in mapped_test_by_id.values():
        for label_id, count in _label_counts(labels).items():
            test_counts[label_id] = test_counts.get(label_id, 0) + count
    return {
        'nominal_train_counts': nominal_train_counts,
        'sampled_train_counts': sampled_train_counts,
        'test_counts': test_counts,
        'test_rows_by_id': {
            int(sample_id): int(len(labels))
            for sample_id, labels in mapped_test_by_id.items()
        },
    }


def load_metadata_payload(metadata_path: str) -> Dict[str, object]:
    try:
        payload = json.loads(read_bytes_handling_gzip(metadata_path).decode('utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Failed to parse metadata JSON: {metadata_path}') from exc
    if not isinstance(payload, dict):
        raise ValueError('Metadata JSON must be an object.')

    dataset = payload.get('dataset')
    if not isinstance(dataset, dict):
        legacy_metadata = payload.get('metadata')
        dataset = dict(legacy_metadata) if isinstance(legacy_metadata, dict) else {}
        payload['dataset'] = dataset

    samples = payload.get('samples')
    if not isinstance(samples, dict):
        samples = {}
        payload['samples'] = samples

    order = samples.get('order')
    if order is None:
        order = payload.get('order')
    if not isinstance(order, list) or not order:
        raise ValueError("Metadata JSON must contain a non-empty 'samples.order' list.")
    if not all(isinstance(item, int) for item in order):
        raise ValueError('Metadata order entries must be integers.')
    if len(set(order)) != len(order):
        raise ValueError('Metadata order entries must be unique.')
    samples['order'] = order
    return _populate_metadata_legacy_aliases(payload)


def _normalize_label_series(labels: pd.Series) -> pd.Series:
    normalized = labels.copy()
    stripped = normalized.astype(str).str.strip()
    mask = normalized.map(_is_non_target_label)
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
        '--data.metadata',
        type=str,
        required=False,
        help="Path to metadata JSON.gz containing samples.order and dataset context.",
    )
    parser.add_argument(
        '--data.import_metadata',
        type=str,
        required=False,
        help='Alias for benchmark-wired import metadata input.',
    )
    parser.add_argument(
        '--num',
        type=int,
        required=True,
        help='1-based index into samples.order for the training sample.',
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
    metadata_path = getattr(args, 'data.metadata') or getattr(
        args, 'data.import_metadata'
    )
    if metadata_path is None:
        raise SystemExit('Either --data.metadata or --data.import_metadata is required.')
    output_dir = args.output_dir
    name = args.name
    num = args.num
    requested_num = num
    test_sample_limit = args.test_sample_limit
    max_workers_arg = args.max_workers

    if not is_tar_archive(raw_path):
        raise ValueError(
            '--data.raw must be a tar/tar.gz archive produced by data_import.'
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_payload = load_metadata_payload(metadata_path)
    dataset_section = metadata_payload.get('dataset')
    if not isinstance(dataset_section, dict):
        dataset_section = {}
    samples_section = metadata_payload.get('samples')
    if not isinstance(samples_section, dict):
        samples_section = {}
    order = samples_section['order']

    dataset_name_raw = dataset_section.get('dataset_name')
    dataset_name: Optional[str] = None
    if isinstance(dataset_name_raw, str) and dataset_name_raw.strip():
        dataset_name = dataset_name_raw.strip()

    sub_sampling_raw = dataset_section.get('sub_sampling', 0)
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
    cache_dir = _archive_cache_dir(raw_path)
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
        archive_counts = _write_sample_archives_from_paths(
            out_dir=out_dir,
            name=name,
            train_id=train_id,
            test_ids=test_ids,
            sample_paths=sample_paths,
            id_to_label=id_to_label,
            preferred_label_col=label_col,
            sub_sampling=sub_sampling,
            cache_dir=cache_dir,
        )
        output_metadata = _copy_json_object(metadata_payload)
        output_dataset = output_metadata.setdefault('dataset', {})
        output_samples = output_metadata.setdefault('samples', {})
        output_labels = output_metadata.setdefault('labels', {})
        output_stages = output_metadata.setdefault('stages', {})

        if dataset_name is not None:
            output_dataset['dataset_name'] = dataset_name

        id_to_label_json = {str(k): v for k, v in sorted(id_to_label.items())}
        label_to_id_json = {value: int(key) for key, value in id_to_label_json.items()}
        output_labels['id_to_label'] = id_to_label_json
        output_labels['label_to_id'] = label_to_id_json
        output_labels['non_target_aliases'] = sorted(UNLABELED_VALUES)
        output_samples['order'] = order

        output_stages['preprocessing'] = {
            'requested_num': int(requested_num),
            'resolved_num': int(num),
            'train_sample_id': int(train_id),
            'test_sample_ids': [int(item) for item in test_ids],
            'train_sample_name': sample_paths[train_id].name,
            'test_sample_names': [sample_paths[item].name for item in test_ids],
            'sub_sampling': int(sub_sampling),
            'test_sample_limit': None if test_sample_limit is None else int(test_sample_limit),
            'max_workers': int(max_workers),
        }
        nominal_train_counts = archive_counts['nominal_train_counts']
        sampled_train_counts = archive_counts['sampled_train_counts']
        test_counts = archive_counts['test_counts']
        test_rows_by_id = archive_counts['test_rows_by_id']
        populations = [
            {
                'id': int(population_id),
                'name': population_name,
                'nominal_train_count': int(
                    nominal_train_counts.get(population_id, 0)
                ),
                'training_support': int(
                    sampled_train_counts.get(population_id, 0)
                ),
                'test_truth_count': int(test_counts.get(population_id, 0)),
                'present_in_training': bool(
                    sampled_train_counts.get(population_id, 0) > 0
                ),
            }
            for population_id, population_name in sorted(id_to_label.items())
        ]
        sampled_rows = int(sum(sampled_train_counts.values()))
        nominal_rows = int(sum(nominal_train_counts.values()))
        total_test_rows = int(sum(test_counts.values()))
        split_audit = {
            'schema_version': '1.0.0',
            'stage': 'preprocessing',
            'identities': {
                'dataset': {
                    'metadata': _copy_json_object(output_dataset),
                    'data_import': _copy_json_object(
                        output_stages.get('data_import', {})
                    ),
                },
                'preprocessing': {
                    'name': name,
                    'parameters': {
                        'requested_fold': int(requested_num),
                        'effective_fold': int(num),
                        'sub_sampling': int(sub_sampling),
                        'test_sample_limit': (
                            None
                            if test_sample_limit is None
                            else int(test_sample_limit)
                        ),
                        'max_workers': int(max_workers),
                    },
                },
            },
            'split': {
                'requested_fold': int(requested_num),
                'effective_fold': int(num),
                'wrapped_fold': {
                    'status': requested_num != num,
                    'reason': (
                        'requested_fold_exceeds_sample_count'
                        if requested_num != num
                        else None
                    ),
                },
                'training_sample': {
                    'id': int(train_id),
                    'name': sample_paths[train_id].name,
                },
                'test_samples': [
                    {'id': int(item), 'name': sample_paths[item].name}
                    for item in test_ids
                ],
            },
            'counts': {
                'training': {
                    'nominal_rows': nominal_rows,
                    'sampled_rows': sampled_rows,
                    'rows_before_filtering': sampled_rows,
                    'rows_after_filtering': sampled_rows,
                    'eligible_rows': int(sampled_rows - sampled_train_counts.get(0, 0)),
                },
                'test': {
                    'rows_before_filtering': total_test_rows,
                    'rows_after_filtering': total_test_rows,
                    'rows_by_sample': [
                        {
                            'id': int(item),
                            'name': sample_paths[item].name,
                            'rows_before_filtering': int(test_rows_by_id[item]),
                            'rows_after_filtering': int(test_rows_by_id[item]),
                        }
                        for item in test_ids
                    ],
                },
            },
            'populations': populations,
            'non_target': {
                'id': 0,
                'name': 'ungated_or_rejection',
                'biological_population': False,
                'nominal_train_count': int(nominal_train_counts.get(0, 0)),
                'sampled_train_count': int(sampled_train_counts.get(0, 0)),
                'final_train_count': int(sampled_train_counts.get(0, 0)),
                'test_truth_count_before_filtering': int(test_counts.get(0, 0)),
                'test_truth_count_after_filtering': int(test_counts.get(0, 0)),
            },
        }
        output_metadata['split_audit'] = split_audit
        _write_metadata(out_dir, name, output_metadata)
        _write_json_gzip(out_dir / f'{name}.split_audit.json.gz', split_audit)
    finally:
        tmp_dir.cleanup()


if __name__ == '__main__':
    main()
