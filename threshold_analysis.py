from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import precision_score, recall_score, f1_score
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# =====================================================
# 1. PATHS AND SETTINGS
# =====================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_DIR / "datasets" / "MVTecAD" / "bottle"
TEST_DIR = DATASET_ROOT / "test"
MASK_ROOT = DATASET_ROOT / "ground_truth"
CHECKPOINT = (
    PROJECT_DIR / "results" / "Patchcore" / "MVTecAD" / "bottle"
    / "v0" / "weights" / "lightning" / "model.ckpt"
)
OUTPUT_DIR = PROJECT_DIR / "detailed_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["good", "broken_large", "broken_small", "contamination"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# These are percentiles of the combined anomaly-map values.
# They are experiment candidates, not universal PatchCore thresholds.
CANDIDATE_PERCENTILES = [90, 95, 97, 98, 99, 99.5, 99.7, 99.9]


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def get_prediction_path(prediction):
    value = getattr(prediction, "image_path", None)
    if value is None and isinstance(prediction, dict):
        value = prediction.get("image_path")
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return Path(str(value)) if value is not None else None


def get_field(prediction, field):
    if isinstance(prediction, dict):
        return prediction[field]
    return getattr(prediction, field)


def resize_mask(mask_path, target_shape):
    height, width = target_shape
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def calculate_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "predicted_defect_pixels": int(np.sum(y_pred)),
        "ground_truth_defect_pixels": int(np.sum(y_true)),
    }


# =====================================================
# 2. CHECK FILES
# =====================================================
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
if not TEST_DIR.exists() or not MASK_ROOT.exists():
    raise FileNotFoundError("MVTec AD bottle test or ground_truth folder not found.")

print("Threshold analysis: MVTec AD Bottle")
print("This is an exploratory threshold comparison on the 83 test images.")
print("Do not treat the best test-set threshold as an unbiased final result.")


# =====================================================
# 3. LOAD PATCHCORE
# =====================================================
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=6,
    coreset_sampling_ratio=0.1,
    precision="float32",
)
engine = Engine(accelerator="cpu", devices=1)


# =====================================================
# 4. COLLECT ANOMALY MAPS AND GROUND-TRUTH MASKS
# =====================================================
all_scores = []
all_ground_truth = []
image_records = []

for category in CATEGORIES:
    category_folder = TEST_DIR / category
    image_paths = sorted(
        p for p in category_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"Skipping empty category: {category}")
        continue

    print(f"\nProcessing {category}: {len(image_paths)} images")

    dataset = PredictDataset(path=category_folder, image_size=(256, 256))
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT),
        return_predictions=True,
    )

    if predictions is None:
        raise RuntimeError(f"No predictions returned for {category}")
    if not isinstance(predictions, (list, tuple)):
        predictions = list(predictions)

    prediction_by_name = {}
    for prediction in predictions:
        pred_path = get_prediction_path(prediction)
        if pred_path is not None:
            prediction_by_name[pred_path.name] = prediction

    for image_path in image_paths:
        prediction = prediction_by_name.get(image_path.name)
        if prediction is None:
            raise RuntimeError(
                f"Could not match prediction for {image_path.name} in {category}"
            )

        anomaly_map = np.squeeze(
            as_numpy(get_field(prediction, "anomaly_map"))
        ).astype(np.float32)

        if anomaly_map.ndim != 2:
            raise ValueError(
                f"Unexpected anomaly map shape for {image_path.name}: "
                f"{anomaly_map.shape}"
            )

        if category == "good":
            gt_mask = np.zeros(anomaly_map.shape, dtype=np.uint8)
        else:
            mask_path = MASK_ROOT / category / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")
            gt_mask = resize_mask(mask_path, anomaly_map.shape)

        if gt_mask.shape != anomaly_map.shape:
            raise ValueError("Ground-truth mask and anomaly map dimensions differ.")

        score_flat = anomaly_map.reshape(-1)
        gt_flat = gt_mask.reshape(-1)

        all_scores.append(score_flat)
        all_ground_truth.append(gt_flat)

        image_records.append({
            "image_name": image_path.name,
            "category": category,
            "scores": score_flat,
            "ground_truth": gt_flat,
        })

        print(
            f"{category}/{image_path.name}: "
            f"GT defect pixels={int(gt_flat.sum())}"
        )


# =====================================================
# 5. BUILD THRESHOLD CANDIDATES
# =====================================================
scores = np.concatenate(all_scores)
ground_truth = np.concatenate(all_ground_truth)

thresholds = np.unique(
    np.percentile(scores, CANDIDATE_PERCENTILES)
)

# Include the current Anomalib predicted-mask result as a reference
# only if it is available from the prior pixel evaluation CSV.
# The threshold sweep itself uses anomaly_map >= threshold.


# =====================================================
# 6. CALCULATE OVERALL METRICS AT EACH THRESHOLD
# =====================================================
threshold_rows = []

for threshold in thresholds:
    predicted = (scores >= threshold).astype(np.uint8)
    metrics = calculate_metrics(ground_truth, predicted)

    threshold_rows.append({
        "threshold": float(threshold),
        "threshold_percentile": float(
            np.mean(scores <= threshold) * 100
        ),
        **metrics,
    })

threshold_df = pd.DataFrame(threshold_rows).sort_values("threshold")
threshold_csv = OUTPUT_DIR / "threshold_analysis_results.csv"
threshold_df.to_csv(threshold_csv, index=False)


# =====================================================
# 7. SELECT BEST TEST-SET F1 (EXPLORATORY ONLY)
# =====================================================
best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
best_threshold = float(best_row["threshold"])

print("\n==============================================")
print("EXPLORATORY THRESHOLD RESULTS")
print("==============================================")
print(threshold_df.to_string(index=False))
print("\nBest threshold on these test images (exploratory only):")
print(f"Threshold: {best_threshold:.8f}")
print(f"Precision: {best_row['precision']:.6f}")
print(f"Recall:    {best_row['recall']:.6f}")
print(f"F1-score:  {best_row['f1']:.6f}")
print(
    "\nReminder: selecting a threshold on the test set makes this "
    "best F1-score unsuitable as an unbiased final test result."
)


# =====================================================
# 8. CATEGORY METRICS AT THE SAME GLOBAL THRESHOLD
# =====================================================
category_rows = []

for category in CATEGORIES:
    category_records = [
        record for record in image_records
        if record["category"] == category
    ]
    if not category_records:
        continue

    category_scores = np.concatenate(
        [record["scores"] for record in category_records]
    )
    category_gt = np.concatenate(
        [record["ground_truth"] for record in category_records]
    )
    category_pred = (category_scores >= best_threshold).astype(np.uint8)

    category_rows.append({
        "category": category,
        "images": len(category_records),
        **calculate_metrics(category_gt, category_pred),
    })

category_df = pd.DataFrame(category_rows)
category_csv = OUTPUT_DIR / "threshold_category_results.csv"
category_df.to_csv(category_csv, index=False)


# =====================================================
# 9. PLOT F1, PRECISION AND RECALL VS THRESHOLD
# =====================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1-score")
ax.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    marker="o",
    label="Precision",
)
ax.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    marker="o",
    label="Recall",
)
ax.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best exploratory threshold = {best_threshold:.4g}",
)
ax.set_xlabel("Anomaly-map threshold")
ax.set_ylabel("Metric")
ax.set_ylim(0, 1.05)
ax.set_title("Pixel Metrics at Different Anomaly-map Thresholds")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()

plot_path = OUTPUT_DIR / "threshold_analysis_plot.png"
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)


# =====================================================
# 10. SAVE A SMALL TEXT SUMMARY
# =====================================================
summary_path = OUTPUT_DIR / "threshold_analysis_summary.txt"
with open(summary_path, "w", encoding="utf-8") as file:
    file.write("PATCHCORE THRESHOLD ANALYSIS - EXPLORATORY TEST\n")
    file.write("Dataset: MVTec AD Bottle, 83 test images\n")
    file.write("Threshold selected on test data; not an unbiased final result.\n\n")
    file.write(f"Best threshold: {best_threshold:.8f}\n")
    file.write(f"Precision: {best_row['precision']:.6f}\n")
    file.write(f"Recall: {best_row['recall']:.6f}\n")
    file.write(f"F1-score: {best_row['f1']:.6f}\n\n")
    file.write("Category metrics at the same threshold:\n")
    file.write(category_df.to_string(index=False))

print("\nSaved files:")
print(threshold_csv)
print(category_csv)
print(plot_path)
print(summary_path)
print("\nThreshold analysis completed.")
