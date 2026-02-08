#!/usr/bin/env python

"""
CLI utility to split CSV sample data into train/test tarballs using an explicit order.

Args:
    --data.raw          Path to a tar.gz archive (or directory) of CSV files.
    --data.order        Path to JSON with {"order": [1, 2, ...]} (1-based sample indices).
    --data.attachments  Optional attachments archive (currently unused).
    --num               1-based index into data.order to pick the training sample.
    --output_dir        Directory where the matrix/label archives will be written.
    --name              Dataset name used for the output filenames.
"""

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fcsparser
import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401

    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False

# Ensure multiprocessing uses spawn to avoid fork-related deadlocks with native extensions.
try:
    import multiprocessing as _mp

    _mp.set_start_method("spawn", force=True)
except Exception:
    # If the start method is already set or unsupported, continue without failing.
    pass


def read_bytes_handling_gzip(path: str) -> bytes:
    """
    Return file contents, transparently handling gzip-compressed files.

    Some inputs may have a .gz suffix even when they are plain text; fall back to
    normal reads if gzip decompression fails.
    """
    try:
        with gzip.open(path, "rb") as fh:
            return fh.read()
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as fh:
            return fh.read()


def parse_fcs_to_dataframe(raw_gz_path: str):
    data_bytes = read_bytes_handling_gzip(raw_gz_path)

    # fcsparser.parse expects a file path; use a temporary file to avoid keeping data on disk.
    with tempfile.NamedTemporaryFile(suffix=".fcs", delete=False) as tmp:
        tmp.write(data_bytes)
        tmp_path = tmp.name

    try:
        _, data = fcsparser.parse(tmp_path, reformat_meta=True)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass  # If cleanup fails, we still want to return the parsed data/error.

    return data


def parse_label_lines(label_text: str, expected_count: int, source: str) -> List[str]:
    labels = [line.strip() for line in label_text.splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {source}.")

    if len(labels) != expected_count:
        raise ValueError(
            f"Label count ({len(labels)}) does not match number of columns ({expected_count})."
        )
    return labels


def detect_label_format(label_path: str, label_text: str) -> str:
    """Return 'txt' or 'xml' based on path suffix or content."""
    suffixes = [s.lower() for s in Path(label_path).suffixes if s.lower() != ".gz"]
    if ".xml" in suffixes:
        return "xml"
    if ".txt" in suffixes:
        return "txt"

    stripped = label_text.lstrip()
    if stripped.startswith("<"):
        return "xml"

    return "txt"


def is_flowjo_workspace(path: str) -> bool:
    suffixes = [s.lower() for s in Path(path).suffixes]
    suffixes = [s for s in suffixes if s not in {".gz", ".zip"}]
    if ".wps" in suffixes or ".wsp" in suffixes:
        return True

    try:
        sample = read_bytes_handling_gzip(path)
        text = sample[:2048].decode("utf-8", errors="ignore").lower()
        if "<workspace" in text and "flowjo" in text:
            return True
    except Exception:
        pass

    return False


def apply_labels(label_gz_path: str, df):
    """Apply labels to DataFrame columns according to the provided rules."""
    try:
        label_text = read_bytes_handling_gzip(label_gz_path).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Unexpected label file format: unable to decode as UTF-8 text."
        ) from exc

    if not label_text.strip():
        raise ValueError(
            "Unexpected label file format: file is empty after decompression."
        )

    label_format = detect_label_format(label_gz_path, label_text)

    if label_format == "xml":
        raise NotImplementedError("XML label handling not implemented.")
    if label_format != "txt":
        raise ValueError("Unexpected label file format.")

    try:
        labels = parse_label_lines(
            label_text, expected_count=df.shape[1], source=label_gz_path
        )
    except ValueError as exc:
        print(
            f"Warning: {exc} Column relabeling skipped; keeping original headers.",
            file=sys.stderr,
        )
        return df
    df.columns = labels
    return df


def collect_fcs_inputs(raw_input: str) -> List[Path]:
    """
    Accept a single FCS path or a directory of FCS files (optionally gzipped) and return a sorted list of paths.
    """
    path = Path(raw_input)
    if path.is_dir():
        candidates = sorted(
            [
                p
                for p in path.iterdir()
                if any(suffix.lower() == ".fcs" for suffix in p.suffixes)
            ]
            + [
                p
                for p in path.iterdir()
                if tuple(s.lower() for s in p.suffixes[-2:]) == (".fcs", ".gz")
            ]
        )
        if not candidates:
            raise FileNotFoundError(f"No FCS files found in directory: {raw_input}")
        return candidates
    if not path.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {raw_input}")
    return [path]


def is_tar_archive(path: Path) -> bool:
    """Return True if the provided path points to a tar (or tar.gz) archive."""
    if not path.is_file():
        return False
    try:
        return tarfile.is_tarfile(path)
    except (OSError, tarfile.TarError):
        return False


@contextmanager
def extract_fcs_from_tar(tar_path: Path) -> Iterable[List[Path]]:
    """
    Extract FCS files from a tar/tar.gz archive into a temporary directory and yield their paths.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    try:
        with tarfile.open(tar_path, mode="r:*") as tar:
            members = [m for m in tar.getmembers() if m.name.lower().endswith(".fcs")]
            if not members:
                raise FileNotFoundError(f"No FCS files found in archive: {tar_path}")
            extracted: List[Path] = []
            for member in members:
                tar.extract(member, path=tmp_dir.name, filter="data")
                extracted.append(Path(tmp_dir.name) / member.name)
        yield sorted(extracted)
    finally:
        tmp_dir.cleanup()


@contextmanager
def prepared_fcs_paths(fcs_paths: Sequence[Path]) -> Iterable[List[Path]]:
    """
    Ensure every FCS is an on-disk uncompressed file so FlowJo parsers can load them.
    Returns a list of usable paths and cleans up any temporary files afterwards.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    prepared: List[Path] = []
    try:
        for fcs_path in fcs_paths:
            suffixes = [s.lower() for s in fcs_path.suffixes]
            if suffixes and suffixes[-1] == ".gz":
                target_name = fcs_path.name
                if target_name.lower().endswith(".gz"):
                    target_name = target_name[: -len(".gz")]
                target_path = Path(tmp_dir.name) / target_name
                with gzip.open(fcs_path, "rb") as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                prepared.append(target_path)
            else:
                prepared.append(fcs_path)
        yield prepared
    finally:
        tmp_dir.cleanup()


@contextmanager
def prepared_fcs_inputs(raw_input: str) -> Iterable[List[Path]]:
    """
    Load FCS inputs from a path that may be a single file, directory, or tar/tar.gz archive.
    Yields ready-to-use uncompressed FCS paths and handles cleanup.
    """
    raw_path = Path(raw_input)
    if raw_path.is_file() and is_tar_archive(raw_path):
        with extract_fcs_from_tar(raw_path) as extracted:
            with prepared_fcs_paths(extracted) as ready:
                yield ready
        return

    fcs_paths = collect_fcs_inputs(raw_input)
    with prepared_fcs_paths(fcs_paths) as ready:
        yield ready


def collect_csv_inputs(raw_input: str) -> List[Path]:
    """
    Accept a single CSV path or a directory of CSV files and return a sorted list of paths.
    """
    path = Path(raw_input)
    if path.is_dir():
        candidates = sorted(
            [p for p in path.iterdir() if p.suffix.lower() == ".csv"]
            + [
                p
                for p in path.iterdir()
                if tuple(s.lower() for s in p.suffixes[-2:]) == (".csv", ".gz")
            ]
        )
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in directory: {raw_input}")
        return candidates
    if not path.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {raw_input}")
    return [path]


@contextmanager
def extract_csv_from_tar(tar_path: Path) -> Iterable[List[Path]]:
    """
    Extract CSV files from a tar/tar.gz archive into a temporary directory and yield their paths.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    try:
        with tarfile.open(tar_path, mode="r:*") as tar:
            members = [
                m
                for m in tar.getmembers()
                if m.name.lower().endswith(".csv") or m.name.lower().endswith(".csv.gz")
            ]
            if not members:
                raise FileNotFoundError(f"No CSV files found in archive: {tar_path}")
            extracted: List[Path] = []
            for member in members:
                tar.extract(member, path=tmp_dir.name, filter="data")
                extracted.append(Path(tmp_dir.name) / member.name)
        yield sorted(extracted, key=lambda p: p.name)
    finally:
        tmp_dir.cleanup()


@contextmanager
def prepared_csv_inputs(raw_input: str) -> Iterable[List[Path]]:
    """
    Load CSV inputs from a path that may be a single file, directory, or tar/tar.gz archive.
    """
    raw_path = Path(raw_input)
    if raw_path.is_file() and is_tar_archive(raw_path):
        with extract_csv_from_tar(raw_path) as extracted:
            yield extracted
        return

    csv_paths = collect_csv_inputs(raw_input)
    yield sorted(csv_paths, key=lambda p: p.name)


def read_csv_dataframe(path: Path) -> pd.DataFrame:
    """Read a CSV or gzipped CSV into a DataFrame."""
    engine = "pyarrow" if _PYARROW_AVAILABLE else None
    read_kwargs = {"engine": engine} if engine else {}
    if path.suffix.lower() == ".gz" or path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip", **read_kwargs)
    return pd.read_csv(path, **read_kwargs)


def read_csv_header(path: Path) -> List[str]:
    if path.suffix.lower() == ".gz" or path.name.lower().endswith(".csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            try:
                return next(reader)
            except StopIteration as exc:
                raise ValueError(f"CSV file has no header row: {path.name}") from exc

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file has no header row: {path.name}") from exc


def _read_csv_column(path: Path, column_name: str) -> pd.Series:
    engine = "pyarrow" if _PYARROW_AVAILABLE else None
    read_kwargs = {"engine": engine} if engine else {}
    if path.suffix.lower() == ".gz" or path.name.lower().endswith(".csv.gz"):
        df = pd.read_csv(path, compression="gzip", usecols=[column_name], **read_kwargs)
    else:
        df = pd.read_csv(path, usecols=[column_name], **read_kwargs)
    return df[column_name]


def _read_csv_row_count(path: Path, first_column: str) -> int:
    engine = "pyarrow" if _PYARROW_AVAILABLE else None
    read_kwargs = {"engine": engine} if engine else {}
    if path.suffix.lower() == ".gz" or path.name.lower().endswith(".csv.gz"):
        df = pd.read_csv(path, compression="gzip", usecols=[first_column], **read_kwargs)
    else:
        df = pd.read_csv(path, usecols=[first_column], **read_kwargs)
    return len(df)


def _load_csv_sample(
    sample_id: int, csv_path: Path, label_col: Optional[str]
) -> Tuple[int, pd.DataFrame, pd.Series]:
    df = read_csv_dataframe(csv_path)
    features, labels = extract_labels_with_column(df, label_col)
    return sample_id, features, labels


def _load_csv_labels_only(
    sample_id: int, csv_path: Path, preferred_label_col: Optional[str]
) -> Tuple[int, pd.Series]:
    header = read_csv_header(csv_path)
    label_col = None
    if preferred_label_col and preferred_label_col in header:
        label_col = preferred_label_col
    else:
        label_col = find_label_column_from_headers(header)

    if label_col is None:
        if not header:
            raise ValueError(f"CSV file has no header row: {csv_path.name}")
        row_count = _read_csv_row_count(csv_path, header[0])
        return sample_id, pd.Series(["unlabeled"] * row_count, name="label")

    labels = _read_csv_column(csv_path, label_col)
    return sample_id, labels


def _collect_sample_label_values(
    sample_id: int, csv_path: Path, preferred_label_col: Optional[str]
) -> Tuple[int, set[str]]:
    header = read_csv_header(csv_path)
    if preferred_label_col and preferred_label_col in header:
        label_col = preferred_label_col
    else:
        label_col = find_label_column_from_headers(header)

    if label_col is None:
        return sample_id, set()

    labels = _read_csv_column(csv_path, label_col)
    normalized = _normalize_label_series(labels)
    values = normalized.dropna().astype(str).str.strip()
    return sample_id, {
        value for value in values if value and value.lower() not in UNLABELED_VALUES
    }


def _flowjo_leaf_gate_paths(
    workspace, sample_id: str
) -> List[Tuple[str, Tuple[str, ...]]]:
    """
    Return (gate_name, gate_path) pairs for leaf gates for the given sample.
    """
    gate_records = [
        (name, tuple(path)) for name, path in workspace.get_gate_ids(sample_id)
    ]
    gate_full_paths = [
        (name, ancestors, ancestors + (name,)) for name, ancestors in gate_records
    ]

    def is_prefix(prefix: Tuple[str, ...], candidate: Tuple[str, ...]) -> bool:
        return len(prefix) <= len(candidate) and candidate[: len(prefix)] == prefix

    leaves: List[Tuple[str, Tuple[str, ...]]] = []
    for name, ancestors, full_path in gate_full_paths:
        has_child = any(
            is_prefix(full_path, other_full) and other_full != full_path
            for _, _, other_full in gate_full_paths
        )
        if not has_child:
            leaves.append((name, ancestors))
    return leaves


def _flowjo_leaf_labels(
    gating_result,
    leaves: Sequence[Tuple[str, Optional[Sequence[str]]]],
    event_count: int,
) -> pd.Series:
    """
    Convert FlowJo gating results into a label Series by assigning the leaf gate name to each event.
    Unassigned events are labeled 'unlabeled'.
    """
    labels = np.full(event_count, "unlabeled", dtype=object)
    for gate_name, gate_path in leaves:
        if hasattr(gating_result, "get_gate_membership"):
            try:
                if gate_path:
                    mask = gating_result.get_gate_membership(
                        gate_name, gate_path=tuple(gate_path)
                    )
                else:
                    mask = gating_result.get_gate_membership(gate_name)
            except TypeError:
                mask = gating_result.get_gate_membership(gate_name)
        elif hasattr(gating_result, "get_population_mask"):
            mask = gating_result.get_population_mask(gate_name)
        else:
            raise RuntimeError(
                "FlowJo gating result does not expose gate membership accessors."
            )
        labels[np.asarray(mask, dtype=bool)] = gate_name
    return pd.Series(labels, name="label")


def label_samples_from_flowjo_workspace(
    workspace_path: str, fcs_paths: Sequence[Path]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Use a FlowJo workspace (.wsp/.wps) to gate a collection of FCS files and emit per-event labels.
    A missing FlowKit dependency raises a clear error.
    """
    try:
        import flowkit as fk  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FlowJo workspace inputs require the 'flowkit' package. "
            "Install it (e.g., pip install flowkit) and re-run this step."
        ) from exc

    try:
        workspace = fk.Workspace(
            workspace_path,
            fcs_samples=[str(p) for p in fcs_paths],
            ignore_missing_files=True,
        )
    except TypeError:
        # Older/newer flowkit versions may not accept ignore_missing_files; fall back.
        workspace = fk.Workspace(
            workspace_path,
            fcs_samples=[str(p) for p in fcs_paths],
        )
    try:
        workspace.analyze_samples(use_mp=False)
    except TypeError:
        workspace.analyze_samples()

    feature_frames: List[pd.DataFrame] = []
    label_frames: List[pd.Series] = []

    sample_ids = workspace.get_sample_ids()
    try:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(sample_ids, desc="FlowJo samples", unit="sample")
    except Exception:
        print(f"Processing {len(sample_ids)} FlowJo samples...", file=sys.stderr)
        iterator = sample_ids

    for sample_id in iterator:
        gating_result = workspace.get_gating_results(sample_id)
        if gating_result is None:
            raise RuntimeError(f"No gating results produced for sample {sample_id}.")

        leaf_gates = _flowjo_leaf_gate_paths(workspace, sample_id)
        if not leaf_gates:
            raise RuntimeError(
                f"No leaf gates found in workspace for sample {sample_id}."
            )

        sample = workspace.get_sample(sample_id)
        if hasattr(sample, "as_dataframe"):
            sample_df = sample.as_dataframe(source="raw")
        elif hasattr(sample, "data"):
            sample_df = pd.DataFrame(sample.data)
        else:
            raise RuntimeError("FlowKit sample object does not expose data accessors.")

        label_series = _flowjo_leaf_labels(
            gating_result=gating_result,
            leaves=leaf_gates,
            event_count=len(sample_df),
        )

        feature_frames.append(sample_df)
        label_frames.append(label_series)

    features_df = pd.concat(feature_frames, ignore_index=True)
    labels = pd.concat(label_frames, ignore_index=True)
    return features_df, labels


@contextmanager
def workspace_materialized(path: str) -> Iterable[str]:
    """
    FlowJo workspaces may be gzipped; materialize to disk if needed and yield the usable path.
    """
    suffixes = [s.lower() for s in Path(path).suffixes]
    if suffixes and suffixes[-1] == ".gz":
        with tempfile.NamedTemporaryFile(
            suffix="".join(suffixes[:-1]) or ".wps", delete=False
        ) as tmp:
            tmp.write(read_bytes_handling_gzip(path))
            tmp_path = tmp.name
        try:
            yield tmp_path
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        yield path


def split_features_and_labels(df) -> Tuple:
    """
    Split the loaded dataframe into features and labels if a label column exists.

    The column named 'label' (case-insensitive) is treated as the target vector.
    Returns (features_df, labels_series_or_None).
    """
    label_col = find_label_column(df)
    if label_col is None:
        print(
            "Warning: no label column found; labeling all rows as 'unlabeled'.",
            file=sys.stderr,
        )
        labels = pd.Series(["unlabeled"] * len(df), name="label")
        return df, labels

    labels = df[label_col]
    features = df.drop(columns=[label_col])
    return features, labels


LABEL_COLUMN_CANDIDATES = (
    "label",
    "population",
    "cell_type",
    "celltype",
    "cluster",
    "cluster_id",
)

UNLABELED_VALUES = {"", "unlabeled", "ungated"}
TAR_GZIP_COMPRESSLEVEL = 1


def find_label_column_from_headers(headers: Sequence[object]) -> Optional[str]:
    lower_map = {str(col).strip().lower(): str(col) for col in headers}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    """Return the most likely label column name based on common conventions."""
    return find_label_column_from_headers(df.columns)


def extract_labels_from_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (features, labels) using a heuristic label column selection."""
    label_col = find_label_column(df)
    if label_col is None:
        labels = pd.Series(["unlabeled"] * len(df), name="label")
        return df, labels
    labels = df[label_col]
    features = df.drop(columns=[label_col])
    return features, labels


def extract_labels_with_column(
    df: pd.DataFrame, label_col: Optional[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (features, labels) using a known label column when available."""
    if label_col and label_col in df.columns:
        labels = df[label_col]
        features = df.drop(columns=[label_col])
        return features, labels
    return extract_labels_from_dataframe(df)


def build_label_key(labels: Sequence[pd.Series]) -> Dict[int, str]:
    """Build a stable id_to_label mapping using 1-indexed ids and 0 for unlabeled."""
    label_set = set()
    for series in labels:
        if series is None:
            continue
        values = series.dropna().astype(str).str.strip()
        for value in values:
            if not value:
                continue
            if value.lower() in UNLABELED_VALUES:
                continue
            label_set.add(value)
    ordered = sorted(label_set)
    return {idx + 1: label for idx, label in enumerate(ordered)}


def build_label_key_from_values(label_values: Iterable[str]) -> Dict[int, str]:
    """Build a stable id_to_label mapping from normalized label values."""
    ordered = sorted(
        {
            value
            for value in label_values
            if value and value.lower() not in UNLABELED_VALUES
        }
    )
    return {idx + 1: label for idx, label in enumerate(ordered)}


def map_labels_to_ints(
    labels: pd.Series, id_to_label: Dict[int, str]
) -> pd.Series:
    """Map string labels to integer ids using id_to_label; unlabeled -> 0."""
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
    return pd.Series(mapped, name="label")


def _sanitize_sample_id(path: Path) -> str:
    name = path.name
    lowered = name.lower()
    while lowered.endswith(".gz") or lowered.endswith(".fcs"):
        if lowered.endswith(".gz"):
            name = name[: -len(".gz")]
            lowered = name.lower()
        if lowered.endswith(".fcs"):
            name = name[: -len(".fcs")]
            lowered = name.lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_.-]", "", name)
    return name


def write_gz_csv(df: pd.DataFrame, path: Path, header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, header=header, compression="gzip")


def write_gz_series(s: pd.Series, path: Path, header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(path, index=False, header=header, compression="gzip")


def write_csv(df: pd.DataFrame, path: Path, header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, header=header)


def write_series_csv(series: pd.Series, path: Path, header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, index=False, header=header)


def _write_label_key(out_dir: Path, name: str, id_to_label: Dict[int, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"id_to_label": {str(k): v for k, v in id_to_label.items()}}
    key_path = out_dir / f"{name}.label_key.json.gz"
    with gzip.open(key_path, "wt") as handle:
        json.dump(payload, handle, indent=2)


def _write_test_archives(
    out_dir: Path, name: str, test_samples: Dict[str, Tuple[pd.DataFrame, pd.Series]]
) -> None:
    matrices_path = out_dir / f"{name}.test.matrices.tar.gz"
    labels_path = out_dir / f"{name}.test.labels.tar.gz"
    with tempfile.TemporaryDirectory() as tmpdir:
        matrix_files: List[Path] = []
        label_files: List[Path] = []
        for sid, (features, labels) in test_samples.items():
            matrix_file = Path(tmpdir) / f"{sid}.matrix.csv.gz"
            label_file = Path(tmpdir) / f"{sid}.labels.csv.gz"
            write_gz_csv(features, matrix_file)
            write_gz_series(labels, label_file)
            matrix_files.append(matrix_file)
            label_files.append(label_file)

        with tarfile.open(
            matrices_path, "w:gz", compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            for path in sorted(matrix_files, key=lambda p: p.name):
                tar.add(path, arcname=path.name)

        with tarfile.open(
            labels_path, "w:gz", compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            for path in sorted(label_files, key=lambda p: p.name):
                tar.add(path, arcname=path.name)


def _write_sample_archives(
    out_dir: Path,
    name: str,
    train_id: int,
    test_ids: Sequence[int],
    samples: Dict[int, Tuple[pd.DataFrame, pd.Series]],
) -> None:
    train_matrix_path = out_dir / f"{name}.train.matrix.tar.gz"
    train_labels_path = out_dir / f"{name}.train.labels.tar.gz"
    test_matrices_path = out_dir / f"{name}.test.matrices.tar.gz"
    test_labels_path = out_dir / f"{name}.test.labels.tar.gz"

    with tempfile.TemporaryDirectory() as tmpdir:
        train_features, train_labels = samples[train_id]
        train_matrix_file = Path(tmpdir) / f"{name}-data-{train_id}.csv"
        train_label_file = Path(tmpdir) / f"{name}-label-{train_id}.csv"
        write_csv(train_features, train_matrix_file)
        write_series_csv(train_labels, train_label_file)

        with tarfile.open(
            train_matrix_path, "w:gz", compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            tar.add(train_matrix_file, arcname=train_matrix_file.name)

        with tarfile.open(
            train_labels_path, "w:gz", compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            tar.add(train_label_file, arcname=train_label_file.name)

        test_matrix_files: List[Path] = []
        test_label_files: List[Path] = []
        for test_id in test_ids:
            features, labels = samples[test_id]
            matrix_file = Path(tmpdir) / f"{name}-data-{test_id}.csv"
            label_file = Path(tmpdir) / f"{name}-label-{test_id}.csv"
            write_csv(features, matrix_file)
            write_series_csv(labels, label_file)
            test_matrix_files.append(matrix_file)
            test_label_files.append(label_file)

        with tarfile.open(
            test_matrices_path, "w:gz", compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            for path in test_matrix_files:
                tar.add(path, arcname=path.name)

        with tarfile.open(
            test_labels_path, "w:gz", compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            for path in test_label_files:
                tar.add(path, arcname=path.name)


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
    train_matrix_path = out_dir / f"{name}.train.matrix.tar.gz"
    train_labels_path = out_dir / f"{name}.train.labels.tar.gz"
    test_matrices_path = out_dir / f"{name}.test.matrices.tar.gz"
    test_labels_path = out_dir / f"{name}.test.labels.tar.gz"

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

        train_matrix_file = Path(tmpdir) / f"{name}-data-{train_id}.csv"
        train_label_file = Path(tmpdir) / f"{name}-label-{train_id}.csv"
        write_csv(train_features, train_matrix_file)
        write_series_csv(mapped_train_labels, train_label_file)

        with tarfile.open(
            train_matrix_path, "w:gz", compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            tar.add(train_matrix_file, arcname=train_matrix_file.name)

        with tarfile.open(
            train_labels_path, "w:gz", compresslevel=TAR_GZIP_COMPRESSLEVEL
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

            matrix_file = Path(tmpdir) / f"{name}-data-{test_id}.csv"
            label_file = Path(tmpdir) / f"{name}-label-{test_id}.csv"
            write_csv(test_features, matrix_file)
            write_series_csv(mapped_test_labels, label_file)
            test_matrix_files.append(matrix_file)
            test_label_files.append(label_file)

        with tarfile.open(
            test_matrices_path, "w:gz", compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            for path in test_matrix_files:
                tar.add(path, arcname=path.name)

        with tarfile.open(
            test_labels_path, "w:gz", compresslevel=TAR_GZIP_COMPRESSLEVEL
        ) as tar:
            for path in test_label_files:
                tar.add(path, arcname=path.name)


def load_order(order_path: str) -> List[int]:
    """Load the order list from a JSON file."""
    try:
        payload = json.loads(read_bytes_handling_gzip(order_path).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse order JSON: {order_path}") from exc

    if not isinstance(payload, dict) or "order" not in payload:
        raise ValueError("Order JSON must be an object with an 'order' key.")
    order = payload["order"]
    if not isinstance(order, list) or not order:
        raise ValueError("Order JSON must contain a non-empty list.")
    if not all(isinstance(item, int) for item in order):
        raise ValueError("Order entries must be integers.")
    if len(set(order)) != len(order):
        raise ValueError("Order entries must be unique.")
    return order


def load_order_payload(order_path: str) -> Dict[str, object]:
    """Load the full order JSON payload, preserving metadata."""
    try:
        payload = json.loads(read_bytes_handling_gzip(order_path).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse order JSON: {order_path}") from exc
    if not isinstance(payload, dict) or "order" not in payload:
        raise ValueError("Order JSON must be an object with an 'order' key.")
    order = payload.get("order")
    if not isinstance(order, list) or not order:
        raise ValueError("Order JSON must contain a non-empty list.")
    if not all(isinstance(item, int) for item in order):
        raise ValueError("Order entries must be integers.")
    if len(set(order)) != len(order):
        raise ValueError("Order entries must be unique.")
    return payload


def _normalize_label_series(labels: pd.Series) -> pd.Series:
    normalized = labels.copy()
    stripped = normalized.astype(str).str.strip()
    lower = stripped.str.lower()
    mask = normalized.isna() | lower.isin(UNLABELED_VALUES)
    normalized = stripped
    normalized.loc[mask] = "unlabeled"
    return normalized


def _subsample_training_data(
    features: pd.DataFrame, labels: pd.Series, target_size: int
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_size <= 0:
        return features, labels
    total = len(labels)
    if target_size >= total:
        print(
            "Warning: sub-sampling size exceeds training set; using all cells.",
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
        fractional = (expected - base).rename("fractional")
        tie_break = counts.rename("count")
        priority = (
            pd.concat([fractional, tie_break], axis=1)
            .sort_values(by=["fractional", "count"], ascending=[False, True])
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


def label_samples_from_flowjo_workspace_by_sample(
    workspace_path: str, fcs_paths: Sequence[Path]
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Return per-sample features/labels using FlowJo workspace gating."""
    try:
        import flowkit as fk  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FlowJo workspace inputs require the 'flowkit' package. "
            "Install it (e.g., pip install flowkit) and re-run this step."
        ) from exc

    try:
        workspace = fk.Workspace(
            workspace_path,
            fcs_samples=[str(p) for p in fcs_paths],
            ignore_missing_files=True,
        )
    except TypeError:
        workspace = fk.Workspace(
            workspace_path,
            fcs_samples=[str(p) for p in fcs_paths],
        )
    try:
        workspace.analyze_samples(use_mp=False)
    except TypeError:
        workspace.analyze_samples()

    per_sample: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    sample_ids = workspace.get_sample_ids()
    try:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(sample_ids, desc="FlowJo samples", unit="sample")
    except Exception:
        print(f"Processing {len(sample_ids)} FlowJo samples...", file=sys.stderr)
        iterator = sample_ids

    for sample_id in iterator:
        gating_result = workspace.get_gating_results(sample_id)
        if gating_result is None:
            raise RuntimeError(f"No gating results produced for sample {sample_id}.")

        leaf_gates = _flowjo_leaf_gate_paths(workspace, sample_id)
        if not leaf_gates:
            raise RuntimeError(
                f"No leaf gates found in workspace for sample {sample_id}."
            )

        sample = workspace.get_sample(sample_id)
        if hasattr(sample, "as_dataframe"):
            sample_df = sample.as_dataframe(source="raw")
        elif hasattr(sample, "data"):
            sample_df = pd.DataFrame(sample.data)
        else:
            raise RuntimeError("FlowKit sample object does not expose data accessors.")

        label_series = _flowjo_leaf_labels(
            gating_result=gating_result,
            leaves=leaf_gates,
            event_count=len(sample_df),
        )

        per_sample[str(sample_id)] = (sample_df, label_series)

    return per_sample


def split_train_test(
    features_df: pd.DataFrame,
    labels: Optional[pd.Series],
    method: str,
    seed: int,
    test_sample_limit: Optional[int] = None,
    test_fraction: float = 0.2,
) -> Tuple[
    Tuple[pd.DataFrame, Optional[pd.Series]], Tuple[pd.DataFrame, Optional[pd.Series]]
]:
    """
    Split features (and labels, when available) into train/test partitions.
    """
    if method != "default":
        raise ValueError(
            f"Unsupported split method '{method}'. Only 'default' is implemented."
        )

    if features_df.empty:
        raise ValueError("No data rows found; cannot perform train/test split.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(features_df))
    rng.shuffle(indices)

    test_size = max(1, int(len(indices) * test_fraction))
    if test_size >= len(indices):
        test_size = 1
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    if train_idx.size == 0:
        # Ensure a non-empty train split.
        train_idx, test_idx = indices[:-1], indices[-1:]

    train_features = features_df.iloc[train_idx]
    test_features = features_df.iloc[test_idx]

    if labels is None:
        train_labels = None
        test_labels = None
    else:
        train_labels = labels.iloc[train_idx]
        test_labels = labels.iloc[test_idx]

    if test_sample_limit is not None:
        if test_sample_limit <= 0:
            raise ValueError("test-sample-limit must be a positive integer.")
        if len(test_features) > test_sample_limit:
            chosen = rng.choice(
                np.arange(len(test_features)),
                size=test_sample_limit,
                replace=False,
            )
            chosen = np.sort(chosen)
            test_features = test_features.iloc[chosen]
            if test_labels is not None:
                test_labels = test_labels.iloc[chosen]

    return (train_features, train_labels), (test_features, test_labels)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess CSV samples into train/test archives."
    )
    parser.add_argument(
        "--data.raw",
        type=str,
        required=True,
        help="Tar.gz archive or directory of CSV samples.",
    )
    parser.add_argument(
        "--data.order",
        type=str,
        required=True,
        help="JSON file containing an 'order' array of 1-based sample indices.",
    )
    parser.add_argument(
        "--data.attachments",
        type=str,
        default=None,
        help="Optional attachments archive (currently unused).",
    )
    parser.add_argument(
        "--num",
        type=int,
        required=True,
        help="1-based index into data.order for the training sample.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the resulting archives.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dataset",
        help="Dataset name used for output filenames.",
    )
    parser.add_argument(
        "--test-sample-limit",
        type=int,
        default=None,
        help="Limit number of test samples (order-based).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of CSV loading workers (default: auto).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None):
    parser = parse_args()
    args = parser.parse_args(argv)

    raw_path = getattr(args, "data.raw")
    order_path = getattr(args, "data.order")
    output_dir = args.output_dir
    name = args.name
    num = args.num
    test_sample_limit = args.test_sample_limit
    max_workers_arg = args.max_workers
    attachments_path = getattr(args, "data.attachments")

    if attachments_path:
        print(
            f"Warning: data attachments are unused ({attachments_path}).",
            file=sys.stderr,
        )

    out_dir = Path(output_dir)
    order_payload = load_order_payload(order_path)
    order = order_payload["order"]
    metadata = order_payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    sub_sampling_raw = metadata.get("sub_sampling", 0)
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
        print(
            "Warning: invalid sub_sampling metadata; defaulting to 0.",
            file=sys.stderr,
        )
        sub_sampling = 0
    if sub_sampling < 0:
        print(
            "Warning: sub_sampling metadata must be non-negative; defaulting to 0.",
            file=sys.stderr,
        )
        sub_sampling = 0

    with prepared_csv_inputs(raw_path) as csv_paths:
        if not csv_paths:
            raise FileNotFoundError(f"No CSV inputs found at {raw_path}")

        csv_paths = sorted(csv_paths, key=lambda p: p.name)
        sample_count = len(csv_paths)

        if set(order) != set(range(1, sample_count + 1)):
            raise ValueError(
                "Order must contain each sample index exactly once (1..n)."
            )
        if len(order) != sample_count:
            raise ValueError("Order length must match number of samples.")
        if num < 1:
            raise ValueError("num must be within 1..n.")
        if num > sample_count:
            wrapped_num = ((num - 1) % sample_count) + 1
            print(
                "Warning: num {} exceeds sample_count {}; using num {}. "
                "Duplicate folds will be filtered in metrics.".format(
                    num, sample_count, wrapped_num
                ),
                file=sys.stderr,
            )
            num = wrapped_num

        train_pos = num - 1
        train_id = order[train_pos]

        max_test = max(sample_count - 1, 0)
        if test_sample_limit is None:
            # Emit a short comment to stdout so CI/logs show the defaulting behavior.
            print("# --test-sample-limit not provided; using all remaining samples")
            test_count = max_test
        else:
            if test_sample_limit <= 0:
                raise ValueError("test-sample-limit must be a positive integer.")
            if test_sample_limit > max_test:
                print(
                    "Warning: test-sample-limit exceeds n-1; using all remaining samples.",
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
                raise ValueError("max-workers must be a positive integer.")
            max_workers = min(max_workers_arg, len(indexed_paths))

        label_values: set[str] = set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, path in indexed_paths:
                futures.append(
                    executor.submit(_collect_sample_label_values, idx, path, label_col)
                )

            for future in as_completed(futures):
                _, sample_values = future.result()
                label_values.update(sample_values)

        id_to_label = build_label_key_from_values(label_values)
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
        _write_label_key(out_dir, name, id_to_label)


if __name__ == "__main__":
    main()
