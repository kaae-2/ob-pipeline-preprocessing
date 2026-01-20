#!/usr/bin/env python

"""
CLI utility to convert gzipped FCS data into gzipped CSV outputs and tar archives.

Args:
    --data.raw      Path to a gz-compressed FCS file OR a directory of FCS files.
    --data.labels   Path to a gz-compressed labels file. Text replaces FCS headers; XML is not supported.
    --output_dir    Directory where the matrix/label CSV files will be written.
    --name          Dataset name used for the output filenames.
    --seed          Random seed used for deterministic train/test splits.
    --method        Train/test split method (only 'default' is supported today).
"""

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fcsparser
import numpy as np
import pandas as pd

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


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    """Return the most likely label column name based on common conventions."""
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def extract_labels_from_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (features, labels) using a heuristic label column selection."""
    label_col = find_label_column(df)
    if label_col is None:
        labels = pd.Series(["unlabeled"] * len(df), name="label")
        return df, labels
    labels = df[label_col]
    features = df.drop(columns=[label_col])
    return features, labels


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
            if value.lower() == "unlabeled":
                continue
            label_set.add(value)
    ordered = sorted(label_set)
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
        if not text or text.lower() == "unlabeled":
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

        with tarfile.open(matrices_path, "w:gz") as tar:
            for path in sorted(matrix_files, key=lambda p: p.name):
                tar.add(path, arcname=path.name)

        with tarfile.open(labels_path, "w:gz") as tar:
            for path in sorted(label_files, key=lambda p: p.name):
                tar.add(path, arcname=path.name)


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
        description="Preprocess gzipped FCS data into CSV."
    )
    parser.add_argument(
        "--data.raw",
        type=str,
        required=True,
        help="Gz-compressed FCS data file.",
    )
    parser.add_argument(
        "--data.labels",
        type=str,
        required=True,
        help="Gz-compressed labels file. Text replaces FCS headers; XML is not supported.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the resulting CSV file.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dataset",
        help="Dataset name used for output filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic train/test splits.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="default",
        help="Train/test split method. Only 'default' is supported.",
    )
    parser.add_argument(
        "--test-sample-limit",
        type=int,
        default=None,
        help="Limit number of test samples (random subset).",
    )
    return parser


def main(argv: Iterable[str] = None):
    parser = parse_args()
    args = parser.parse_args(argv)

    raw_path = getattr(args, "data.raw")
    label_path = getattr(args, "data.labels")
    output_dir = args.output_dir
    name = args.name
    seed = args.seed
    method = args.method
    test_sample_limit = args.test_sample_limit

    out_dir = Path(output_dir)
    per_sample: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}

    with prepared_fcs_inputs(raw_path) as ready_fcs:
        if not ready_fcs:
            raise FileNotFoundError(f"No FCS inputs found at {raw_path}")

        if label_path and is_flowjo_workspace(label_path):
            with workspace_materialized(label_path) as workspace_path:
                flowjo_samples = label_samples_from_flowjo_workspace_by_sample(
                    workspace_path, ready_fcs
                )
            for sample_id, (features, labels) in flowjo_samples.items():
                per_sample[_sanitize_sample_id(Path(sample_id))] = (features, labels)
        else:
            for fcs_path in ready_fcs:
                data_df = parse_fcs_to_dataframe(str(fcs_path))
                if label_path:
                    data_df = apply_labels(label_path, data_df)
                features, labels = extract_labels_from_dataframe(data_df)
                per_sample[_sanitize_sample_id(fcs_path)] = (features, labels)

    samples = sorted(per_sample.keys())
    label_series = [labels for _, labels in per_sample.values()]
    id_to_label = build_label_key(label_series)

    mapped_samples: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for sid, (features, labels) in per_sample.items():
        mapped_samples[sid] = (features, map_labels_to_ints(labels, id_to_label))

    if len(samples) == 1:
        feats, labs = mapped_samples[samples[0]]
        (train_feats, train_labels), (test_feats, test_labels) = split_train_test(
            feats, labs, method=method, seed=seed
        )
        if train_labels is None or test_labels is None:
            raise ValueError("Expected labels for single-sample split.")

        write_gz_csv(train_feats, out_dir / f"{name}.train.matrix.csv.gz")
        write_gz_series(train_labels, out_dir / f"{name}.train.labels.csv.gz")
        _write_test_archives(out_dir, name, {samples[0]: (test_feats, test_labels)})
        _write_label_key(out_dir, name, id_to_label)
        return

    rng = np.random.default_rng(seed)
    chosen_train = rng.choice(samples)

    remaining = [sid for sid in samples if sid != chosen_train]
    if test_sample_limit is not None:
        if test_sample_limit <= 0:
            raise ValueError("test-sample-limit must be a positive integer.")
        if len(remaining) > test_sample_limit:
            remaining = sorted(
                rng.choice(remaining, size=test_sample_limit, replace=False)
            )

    test_samples: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for sid in remaining:
        test_samples[sid] = mapped_samples[sid]

    train_feats, train_labels = mapped_samples[chosen_train]
    write_gz_csv(train_feats, out_dir / f"{name}.train.matrix.csv.gz")
    write_gz_series(train_labels, out_dir / f"{name}.train.labels.csv.gz")

    if not test_samples:
        test_samples = {"empty": (pd.DataFrame(), pd.Series(dtype=int))}

    _write_test_archives(out_dir, name, test_samples)
    _write_label_key(out_dir, name, id_to_label)


if __name__ == "__main__":
    main()
