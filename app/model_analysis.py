"""
Dataset Epidemiology, Multi-Label Statistics, and Model Error Analysis.

Added by Nishant Shukla for academic submission in October 2026.

Provides offline analytical tools to inspect the NIH ChestX-ray14 dataset
characteristics, disease co-occurrence matrices, class imbalance distributions,
and multi-label evaluation metrics (Precision, Recall, F1, Hamming Loss).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from app.config import BASE_DIR

CSV_DATA_ENTRY = os.path.join(BASE_DIR, "model_train", "Data_Entry_2017.csv")
CSV_BBOX = os.path.join(BASE_DIR, "model_train", "BBox_List_2017.csv")

ALL_14_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]


def load_dataset_metadata(csv_path: str = CSV_DATA_ENTRY, nrows: int | None = None) -> pd.DataFrame:
    """Safely loads NIH ChestX-ray14 dataset entry metadata."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path, nrows=nrows)
    return df


def calculate_label_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates per-label counts and prevalence percentage across the dataset."""
    total_images = len(df)
    labels_series = df["Finding Labels"].str.split("|").explode()
    counts = labels_series.value_counts()

    stats = []
    for label, count in counts.items():
        stats.append({
            "Label": label,
            "Count": count,
            "Prevalence (%)": round((count / total_images) * 100, 2)
        })
    return pd.DataFrame(stats)


def calculate_cooccurrence_matrix(df: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    """
    Computes symmetric co-occurrence frequencies between disease labels.
    """
    target_labels = labels or ALL_14_LABELS
    matrix = pd.DataFrame(0, index=target_labels, columns=target_labels)

    for finding_str in df["Finding Labels"]:
        present = set(finding_str.split("|"))
        active = [l for l in target_labels if l in present]
        for i in range(len(active)):
            for j in range(len(active)):
                matrix.loc[active[i], active[j]] += 1

    return matrix


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> dict:
    """
    Computes multi-label classification metrics across binary prediction arrays.
    y_true: binary matrix (N, num_labels)
    y_pred: binary matrix (N, num_labels)
    """
    target_labels = labels or ALL_14_LABELS
    num_samples, num_labels = y_true.shape

    per_class = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for idx, label in enumerate(target_labels[:num_labels]):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))

        precision = tp / max(tp + fp, 1e-6)
        recall = tp / max(tp + fn, 1e-6)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
        specificity = tn / max(tn + fp, 1e-6)

        per_class[label] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "specificity": round(float(specificity), 4),
            "support": int(np.sum(yt == 1)),
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    # Macro & Micro metrics
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    macro_precision = np.mean([v["precision"] for v in per_class.values()])
    macro_recall = np.mean([v["recall"] for v in per_class.values()])

    micro_precision = total_tp / max(total_tp + total_fp, 1e-6)
    micro_recall = total_tp / max(total_tp + total_fn, 1e-6)
    micro_f1 = 2 * (micro_precision * micro_recall) / max(micro_precision + micro_recall, 1e-6)

    # Multi-label Hamming loss (fraction of incorrect label predictions)
    hamming_loss = np.mean(y_true != y_pred)

    # Subset accuracy (exact match)
    subset_acc = np.mean(np.all(y_true == y_pred, axis=1))

    return {
        "per_class": per_class,
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "hamming_loss": round(float(hamming_loss), 4),
        "subset_accuracy": round(float(subset_acc), 4),
    }


def format_dataset_summary(csv_path: str = CSV_DATA_ENTRY, sample_limit: int = 10000) -> str:
    """Generates an ASCII text summary of dataset demographics and top findings."""
    df = load_dataset_metadata(csv_path, nrows=sample_limit)
    total = len(df)
    unique_patients = df["Patient ID"].nunique() if "Patient ID" in df else "N/A"
    
    prev_df = calculate_label_prevalence(df)

    lines = [
        "=" * 65,
        "  NIH ChestX-ray14 Dataset Summary & Epidemiological Statistics",
        "=" * 65,
        f"Total Analyzed Images:    {total}",
        f"Unique Patient Cohort:    {unique_patients}",
    ]
    if "Patient Gender" in df:
        gender_counts = df["Patient Gender"].value_counts().to_dict()
        lines.append(f"Gender Distribution:      {gender_counts}")
    if "View Position" in df:
        view_counts = df["View Position"].value_counts().to_dict()
        lines.append(f"View Positions:           {view_counts}")

    lines += [
        "-" * 65,
        f"{'Condition':<25} | {'Cases':<10} | {'Prevalence'}",
        "-" * 65,
    ]
    for _, row in prev_df.head(15).iterrows():
        lines.append(f"{row['Label']:<25} | {row['Count']:<10} | {row['Prevalence (%)']}%")
    lines.append("=" * 65)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 Analysis Suite")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary")
    parser.add_argument("--limit", type=int, default=10000, help="Row limit for metadata read")
    args = parser.parse_args()

    if args.summary:
        print(format_dataset_summary(sample_limit=args.limit))
    else:
        print("Run with --summary to inspect dataset statistics.")
