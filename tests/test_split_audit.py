from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / 'data_preprocessing.py'


def write_fixture(root: Path, samples: dict[str, list[tuple[float, str]]], **dataset) -> tuple[Path, Path]:
    source_dir = root / 'source'
    source_dir.mkdir()
    for sample_name, rows in samples.items():
        with (source_dir / sample_name).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['marker', 'label'])
            writer.writerows(rows)

    archive_path = root / 'raw.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for path in sorted(source_dir.iterdir()):
            archive.add(path, arcname=path.name)

    metadata_path = root / 'metadata.json.gz'
    payload = {
        'schema_version': 1,
        'dataset': {
            'dataset_name': 'fixture',
            'sub_sampling': dataset.pop('sub_sampling', 0),
            'transformation_cofactor': 5,
            'potential_batches': 1,
            **dataset,
        },
        'samples': {
            'order': list(range(1, len(samples) + 1)),
            'sample_names': sorted(samples),
            'sample_count': len(samples),
        },
        'labels': {},
        'stages': {'data_import': {'seed': 42}},
    }
    with gzip.open(metadata_path, 'wt', encoding='utf-8') as handle:
        json.dump(payload, handle)
    return archive_path, metadata_path


def run_preprocessing(root: Path, archive_path: Path, metadata_path: Path, fold: int = 1) -> Path:
    output_dir = root / 'output'
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--data.raw',
            str(archive_path),
            '--data.metadata',
            str(metadata_path),
            '--num',
            str(fold),
            '--max-workers',
            '1',
            '--output_dir',
            str(output_dir),
            '--name',
            'fixture',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_dir


def read_json_gzip(path: Path) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


class SplitAuditCliTests(unittest.TestCase):
    def test_emitted_audit_reports_population_missing_from_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw, metadata = write_fixture(
                root,
                {
                    'sample-a.csv': [(1, 'Alpha'), (2, 'Alpha'), (3, 'ungated')],
                    'sample-b.csv': [(4, 'Beta'), (5, 'Beta')],
                },
            )
            output = run_preprocessing(root, raw, metadata)

            emitted_metadata = read_json_gzip(output / 'fixture.metadata.json.gz')
            emitted_sidecar = read_json_gzip(output / 'fixture.split_audit.json.gz')
            audit = emitted_metadata['split_audit']

            self.assertEqual(emitted_sidecar, audit)
            self.assertEqual(audit['schema_version'], '1.0.0')
            self.assertEqual(audit['split']['training_sample'], {'id': 1, 'name': 'sample-a.csv'})
            self.assertEqual(
                audit['split']['test_samples'],
                [{'id': 2, 'name': 'sample-b.csv'}],
            )
            self.assertEqual(audit['counts']['training']['nominal_rows'], 3)
            self.assertEqual(audit['counts']['training']['eligible_rows'], 2)
            populations = {item['name']: item for item in audit['populations']}
            self.assertEqual(populations['Beta']['nominal_train_count'], 0)
            self.assertEqual(populations['Beta']['training_support'], 0)
            self.assertEqual(populations['Beta']['test_truth_count'], 2)
            self.assertFalse(populations['Beta']['present_in_training'])
            self.assertFalse(audit['non_target']['biological_population'])

    def test_emitted_audit_explains_wrapped_fold_for_small_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw, metadata = write_fixture(
                root,
                {
                    'sample-a.csv': [(1, 'Alpha')],
                    'sample-b.csv': [(2, 'Beta')],
                },
            )
            output = run_preprocessing(root, raw, metadata, fold=5)

            audit = read_json_gzip(output / 'fixture.metadata.json.gz')['split_audit']

            self.assertEqual(audit['split']['requested_fold'], 5)
            self.assertEqual(audit['split']['effective_fold'], 1)
            self.assertEqual(
                audit['split']['wrapped_fold'],
                {
                    'status': True,
                    'reason': 'requested_fold_exceeds_sample_count',
                },
            )
            self.assertEqual(audit['split']['training_sample']['id'], 1)
            self.assertEqual([item['id'] for item in audit['split']['test_samples']], [2])

    def test_emitted_audit_separates_nominal_sampled_and_eligible_training_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw, metadata = write_fixture(
                root,
                {
                    'sample-a.csv': [
                        (1, 'Alpha'),
                        (2, 'Alpha'),
                        (3, 'Alpha'),
                        (4, 'Beta'),
                        (5, 'ungated'),
                        (6, 'ungated'),
                    ],
                    'sample-b.csv': [(7, 'Alpha')],
                },
                sub_sampling=4,
            )
            output = run_preprocessing(root, raw, metadata)

            audit = read_json_gzip(output / 'fixture.split_audit.json.gz')
            training = audit['counts']['training']
            populations = {item['name']: item for item in audit['populations']}

            self.assertEqual(training['nominal_rows'], 6)
            self.assertEqual(training['sampled_rows'], 4)
            self.assertEqual(training['eligible_rows'], 3)
            self.assertEqual(populations['Alpha']['nominal_train_count'], 3)
            self.assertEqual(populations['Alpha']['training_support'], 2)
            self.assertEqual(populations['Beta']['nominal_train_count'], 1)
            self.assertEqual(populations['Beta']['training_support'], 1)
            self.assertEqual(audit['non_target']['nominal_train_count'], 2)
            self.assertEqual(audit['non_target']['sampled_train_count'], 1)

    def test_numeric_zero_is_non_biological_ungated_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw, metadata = write_fixture(
                root,
                {
                    'sample-a.csv': [(1, '0'), (2, '0.0'), (3, 'Alpha')],
                    'sample-b.csv': [(4, 'Beta')],
                },
            )
            output = run_preprocessing(root, raw, metadata)

            audit = read_json_gzip(output / 'fixture.metadata.json.gz')['split_audit']

            self.assertEqual(
                [item['name'] for item in audit['populations']],
                ['Alpha', 'Beta'],
            )
            self.assertEqual(audit['counts']['training']['eligible_rows'], 1)
            self.assertEqual(audit['non_target']['nominal_train_count'], 2)

    def test_single_sample_cohort_emits_an_empty_ordered_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw, metadata = write_fixture(
                root,
                {'only-sample.csv': [(1, 'Alpha'), (2, 'ungated')]},
            )
            output = run_preprocessing(root, raw, metadata, fold=5)

            audit = read_json_gzip(output / 'fixture.split_audit.json.gz')

            self.assertEqual(audit['split']['effective_fold'], 1)
            self.assertTrue(audit['split']['wrapped_fold']['status'])
            self.assertEqual(audit['split']['test_samples'], [])
            self.assertEqual(audit['counts']['test']['rows_before_filtering'], 0)
            self.assertEqual(audit['counts']['test']['rows_after_filtering'], 0)
            with tarfile.open(output / 'fixture.test.labels.tar.gz', 'r:*') as archive:
                self.assertEqual(archive.getmembers(), [])

    def test_duplicate_sample_ids_in_metadata_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw, metadata = write_fixture(
                root,
                {
                    'sample-a.csv': [(1, 'Alpha')],
                    'sample-b.csv': [(2, 'Beta')],
                },
            )
            payload = read_json_gzip(metadata)
            payload['samples']['order'] = [1, 1]
            with gzip.open(metadata, 'wt', encoding='utf-8') as handle:
                json.dump(payload, handle)

            with self.assertRaises(subprocess.CalledProcessError):
                run_preprocessing(root, raw, metadata)
            self.assertFalse((root / 'output' / 'fixture.metadata.json.gz').exists())

    def test_duplicate_sample_members_in_raw_archive_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw, metadata = write_fixture(
                root,
                {
                    'sample-a.csv': [(1, 'Alpha')],
                    'sample-b.csv': [(2, 'Beta')],
                },
            )
            sample = root / 'source' / 'sample-a.csv'
            with tarfile.open(raw, 'w:gz') as archive:
                archive.add(sample, arcname='sample-a.csv')
                archive.add(sample, arcname='sample-a.csv')

            with self.assertRaises(subprocess.CalledProcessError):
                run_preprocessing(root, raw, metadata)

    def test_json_manifests_have_deterministic_bytes_and_key_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            outputs = []
            for fixture_name in ('first', 'second'):
                root = parent / fixture_name
                root.mkdir()
                raw, metadata = write_fixture(
                    root,
                    {
                        'sample-a.csv': [(1, 'Alpha'), (2, 'ungated')],
                        'sample-b.csv': [(3, 'Beta')],
                    },
                )
                outputs.append(run_preprocessing(root, raw, metadata))

            first = outputs[0] / 'fixture.split_audit.json.gz'
            second = outputs[1] / 'fixture.split_audit.json.gz'
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with gzip.open(first, 'rt', encoding='utf-8') as handle:
                serialized = handle.read()
            self.assertLess(serialized.index('"counts"'), serialized.index('"identities"'))


if __name__ == '__main__':
    unittest.main()
