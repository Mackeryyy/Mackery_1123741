
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from anomalib.data import PredictDataset
from anomalib.models import Patchcore
from anomalib.engine import Engine


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

TEST_DIR = PROJECT_DIR / "datasets" / "MVTecAD" / "bottle" / "test"

CHECKPOINT = (
    PROJECT_DIR
    / "results"
    / "Patchcore"
    / "MVTecAD"
    / "bottle"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

OUTPUT_DIR = PROJECT_DIR / "detailed_evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

CATEGORIES = [
    "good",
    "broken_large",
    "broken_small",
    "contamination",
]


# ==========================================
# 2. LOAD MODEL
# ==========================================

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=6,
    coreset_sampling_ratio=0.1,
    precision="float32",
)

engine = Engine(
    accelerator="cpu",
    devices=1,
)

print("Loading PatchCore checkpoint...")


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def first_value(value):
    """Convert a prediction field into one Python value."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()

    value = np.asarray(value)
    return value.reshape(-1)[0].item()


def prediction_value(prediction, field):
    """Read a field from an Anomalib prediction."""
    if isinstance(prediction, dict):
        return prediction[field]
    return getattr(prediction, field)


# ==========================================
# 4. PREDICT ALL TEST IMAGES
# ==========================================

records = []

for category in CATEGORIES:

    folder = TEST_DIR / category

    image_paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
    )

    if not image_paths:
        print(f"No images found in {category}")
        continue

    print(f"\nEvaluating {category}: {len(image_paths)} images")

    dataset = PredictDataset(
        path=folder,
        image_size=(256, 256),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT),
        return_predictions=True,
    )

    if predictions is None:
        raise RuntimeError(
            f"No predictions returned for category: {category}"
        )

    # Handle a single prediction as well as a list of predictions.
    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions]

    for prediction in predictions:

        image_path_value = prediction_value(
            prediction, "image_path"
        )

        if isinstance(image_path_value, (list, tuple)):
            image_path_value = image_path_value[0]

        if hasattr(image_path_value, "item"):
            image_path_value = image_path_value.item()

        image_path = Path(str(image_path_value))

        pred_label = int(first_value(
            prediction_value(prediction, "pred_label")
        ))

        pred_score = float(first_value(
            prediction_value(prediction, "pred_score")
        ))

        # Ground truth:
        # good = 0, all defect categories = 1
        true_label = 0 if category == "good" else 1

        records.append({
            "image_name": image_path.name,
            "defect_type": category,
            "true_label": true_label,
            "predicted_label": pred_label,
            "anomaly_score": pred_score,
            "actual_result": (
                "Normal" if true_label == 0 else "Defective"
            ),
            "predicted_result": (
                "Normal" if pred_label == 0 else "Defective"
            ),
        })

        print(
            f"{image_path.name}: "
            f"Actual={true_label}, "
            f"Predicted={pred_label}, "
            f"Score={pred_score:.4f}"
        )


# ==========================================
# 5. SAVE INDIVIDUAL PREDICTIONS
# ==========================================

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("No predictions were collected.")

csv_path = OUTPUT_DIR / "all_83_predictions.csv"
df.to_csv(csv_path, index=False)

print("\nPrediction CSV saved:", csv_path)


# ==========================================
# 6. CLASSIFICATION METRICS
# ==========================================

y_true = df["true_label"]
y_pred = df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

print("\n====================================")
print("IMAGE CLASSIFICATION PERFORMANCE")
print("====================================")

print(f"Total images: {len(df)}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Defective"],
    yticklabels=["Normal", "Defective"],
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("PatchCore - Bottle Confusion Matrix")
plt.tight_layout()

cm_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300)
plt.close()

print("Confusion matrix saved:", cm_path)


# ==========================================
# 8. PERFORMANCE BY DEFECT TYPE
# ==========================================

summary = []

for category in CATEGORIES:

    subset = df[df["defect_type"] == category]

    if subset.empty:
        continue

    correct = (
        subset["true_label"] == subset["predicted_label"]
    ).sum()

    total = len(subset)

    detection_rate = correct / total

    summary.append({
        "Defect Type": category,
        "Total Images": total,
        "Correct Predictions": int(correct),
        "Detection Rate": detection_rate,
        "Average Anomaly Score": subset["anomaly_score"].mean(),
    })

summary_df = pd.DataFrame(summary)

print("\n====================================")
print("PERFORMANCE BY DEFECT TYPE")
print("====================================")

print(summary_df.to_string(index=False))

summary_path = OUTPUT_DIR / "defect_type_summary.csv"
summary_df.to_csv(summary_path, index=False)


# ==========================================
# 9. DEFECT TYPE CHART
# ==========================================

plt.figure(figsize=(9, 5))

plt.bar(
    summary_df["Defect Type"],
    summary_df["Detection Rate"] * 100,
)

plt.ylabel("Correct Classification (%)")
plt.xlabel("Bottle Category")
plt.title("PatchCore Performance by Defect Type")
plt.ylim(0, 105)
plt.xticks(rotation=15)
plt.tight_layout()

chart_path = OUTPUT_DIR / "defect_type_performance.png"
plt.savefig(chart_path, dpi=300)
plt.close()


# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n====================================")
print("EVALUATION COMPLETED")
print("====================================")

print("Images evaluated:", len(df))
print("CSV:", csv_path)
print("Confusion matrix:", cm_path)
print("Defect summary:", summary_path)
print("Performance chart:", chart_path)
print("\nAll evaluation files are in:")
print(OUTPUT_DIR)
