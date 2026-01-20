#!/usr/bin/env bash
set -euo pipefail

# Run data_preprocessing.py with the requested parameters.
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
out_dir="${script_dir}/out/data/data_import/preprocessing/data_preprocessing/default"

train_matrix="${out_dir}/data_import.train.matrix.csv.gz"
train_labels="${out_dir}/data_import.train.labels.csv.gz"
test_matrices="${out_dir}/data_import.test.matrices.tar.gz"
test_labels="${out_dir}/data_import.test.labels.tar.gz"
label_key="${out_dir}/data_import.label_key.json.gz"

if [[ -f "$train_matrix" && -f "$train_labels" && -f "$test_matrices" && -f "$test_labels" && -f "$label_key" ]]; then
  echo "Preprocessing outputs already present; skipping regenerate."
else
  rm -f "${out_dir}/data_import."*.matrix.gz \
        "${out_dir}/data_import."*.true_labels*.gz \
        "${out_dir}/data_import.test.matrix.csv.gz" \
        "${out_dir}/data_import.test.labels.csv.gz" \
        "${out_dir}/data_import.test.matrices.tar.gz" \
        "${out_dir}/data_import.test.labels.tar.gz" \
        "${out_dir}/data_import.train.matrix.csv.gz" \
        "${out_dir}/data_import.train.labels.csv.gz" \
        "${out_dir}/data_import.label_key.json.gz"

  (cd "$repo_root" && python "preprocessing/data_preprocessing.py" \
    --name "data_import" \
    --output_dir "$out_dir" \
    --data.raw "${repo_root}/datasets/data/covid" \
    --data.labels "${repo_root}/datasets/attachments/01-May-2020_Human_COVID_analysis_template.wsp" \
    --test-sample-limit 5)
fi

# Create gz-suffixed symlinks where models/dgcytof expects them
repo_root="$(cd "$script_dir/.." && pwd)"
src_dir="$out_dir"
model_out_dir="${repo_root}/models/dgcytof/out/data/data_preprocessing/default"

mkdir -p "$model_out_dir"
ln -sf "${src_dir}/data_import.train.matrix.csv.gz" \
      "${model_out_dir}/data_preprocessing.train.matrix.csv.gz"
ln -sf "${src_dir}/data_import.train.labels.csv.gz" \
      "${model_out_dir}/data_preprocessing.train.labels.csv.gz"
ln -sf "${src_dir}/data_import.test.matrices.tar.gz" \
      "${model_out_dir}/data_preprocessing.test.matrices.tar.gz"
ln -sf "${src_dir}/data_import.test.labels.tar.gz" \
      "${model_out_dir}/data_preprocessing.test.labels.tar.gz"
ln -sf "${src_dir}/data_import.label_key.json.gz" \
      "${model_out_dir}/data_preprocessing.label_key.json.gz"

# Create relative symlinks so metrics runner can find outputs
metrics_pred_dir="${repo_root}/metrics/out/data/analysis/default/dgcytof"
metrics_labels_dir="${repo_root}/metrics/out/data/data_preprocessing/default"

mkdir -p "$metrics_pred_dir" "$metrics_labels_dir"

# Use relative symlinks to keep the repo relocatable
if command -v realpath >/dev/null 2>&1; then
  rel_pred="$(realpath --relative-to="$metrics_pred_dir" "${repo_root}/models/dgcytof/out/data/analysis/default/dgcytof/dgcytof_predicted_labels.tar.gz")"
  rel_labels="$(realpath --relative-to="$metrics_labels_dir" "${src_dir}/data_import.test.labels.tar.gz")"
  ln -sf "$rel_pred" "$metrics_pred_dir/dgcytof_predicted_labels.tar.gz"
  ln -sf "$rel_labels" "$metrics_labels_dir/data_preprocessing_labels.tar.gz"
else
  ln -sf "${repo_root}/models/dgcytof/out/data/analysis/default/dgcytof/dgcytof_predicted_labels.tar.gz" \
        "$metrics_pred_dir/dgcytof_predicted_labels.tar.gz"
  ln -sf "${src_dir}/data_import.test.labels.tar.gz" \
        "$metrics_labels_dir/data_preprocessing_labels.tar.gz"
fi
