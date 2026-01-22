#!/usr/bin/env bash
set -euo pipefail

# Run data_preprocessing.py with the requested parameters.
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
out_dir="${script_dir}/out/data/data_import/preprocessing/data_preprocessing/default"
dataset_name="FR-FCM-Z2KP_healthy_final"

train_matrix="${out_dir}/data_import.train.matrix.tar.gz"
train_labels="${out_dir}/data_import.train.labels.tar.gz"
test_matrices="${out_dir}/data_import.test.matrices.tar.gz"
test_labels="${out_dir}/data_import.test.labels.tar.gz"
label_key="${out_dir}/data_import.label_key.json.gz"

rm -f "${out_dir}/data_import."*.matrix.gz \
      "${out_dir}/data_import."*.true_labels*.gz \
      "${out_dir}/data_import.test.matrix.csv.gz" \
      "${out_dir}/data_import.test.labels.csv.gz" \
      "${out_dir}/data_import.test.matrices.tar.gz" \
      "${out_dir}/data_import.test.labels.tar.gz" \
      "${out_dir}/data_import.train.matrix.tar.gz" \
      "${out_dir}/data_import.train.labels.tar.gz" \
      "${out_dir}/data_import.label_key.json.gz"

(cd "$repo_root" && python "preprocessing/data_preprocessing.py" \
  --name "data_import" \
  --output_dir "$out_dir" \
  --data.raw "${script_dir}/out/data/data_import/${dataset_name}.data.gz" \
  --data.order "${script_dir}/out/data/data_import/${dataset_name}.order.json.gz" \
  --data.attachments "${script_dir}/out/data/data_import/${dataset_name}.attachments.gz" \
  --num 1 \
  --test-sample-limit 5)

# Symlink output folders to avoid per-file links
repo_root="$(cd "$script_dir/.." && pwd)"
src_dir="$out_dir"
model_out_dir="${repo_root}/models/dgcytof/out/data/data_preprocessing/default"

mkdir -p "$(dirname "$model_out_dir")"
if [[ -e "$model_out_dir" && ! -L "$model_out_dir" ]]; then
  rm -rf "$model_out_dir"
fi
ln -sfn "$src_dir" "$model_out_dir"

# Create folder symlinks so metrics runner can find outputs
metrics_pred_dir="${repo_root}/metrics/out/data/analysis/default/dgcytof"
metrics_labels_dir="${repo_root}/metrics/out/data/data_preprocessing/default"
pred_src_dir="${repo_root}/models/dgcytof/out/data/analysis/default/dgcytof"

mkdir -p "$(dirname "$metrics_pred_dir")" "$(dirname "$metrics_labels_dir")"
if [[ -e "$metrics_pred_dir" && ! -L "$metrics_pred_dir" ]]; then
  rm -rf "$metrics_pred_dir"
fi
if [[ -e "$metrics_labels_dir" && ! -L "$metrics_labels_dir" ]]; then
  rm -rf "$metrics_labels_dir"
fi
ln -sfn "$pred_src_dir" "$metrics_pred_dir"
ln -sfn "$src_dir" "$metrics_labels_dir"
