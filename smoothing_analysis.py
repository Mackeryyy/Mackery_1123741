from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# =========================================================
# 1. SETTINGS
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_DIR / "datasets" / "MVTecAD" / "bottle"
TEST_DIR = DATASET_ROOT / "test"
GROUND_TRUTH_DIR = DATASET_ROOT / "ground_truth"

CHECKPOINT = (
    PROJECT_DIR / "results" / "Patchcore" / "MVTecAD" / "bottle"
    / "v0" / "weights" / "lightning" / "model.ckpt"
)

OUTPUT_DIR = PROJECT_DIR / "detailed_evaluation" / "smoothing_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["good", "broken_large", "broken_small", "contamination"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# Gaussian sigma values, measured in pixels of the 256 x 256 anomaly map.
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]

# Thresholds are selected as percentiles of each method's combined map values.
# This makes the comparison less sensitive to smoothing changing map amplitudes.
PERCENTILES = [90, 95, 97, 98, 99, 99.5]

IMAGE_SIZE = (256, 256)


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def get_field(prediction, name):
    if isinstance(prediction, dict):
        return prediction[name]
    return getattr(prediction, name)


def get_prediction_path(prediction):
    value = get_field(prediction, "image_path")
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return Path(str(value))


def to_2d_map(value):
    array = np.squeeze(to_numpy(value)).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D anomaly map, got {array.shape}")
    return array


def load_ground_truth_mask(mask_path, shape):
    height, width = shape
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def calculate_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(np.uint8)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "predicted_defect_pixels": int(y_pred.sum()),
        "ground_truth_defect_pixels": int(y_true.sum()),
    }


# =========================================================
# 3. CHECK INPUTS
# =========================================================
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

if not TEST_DIR.exists() or not GROUND_TRUTH_DIR.exists():
    raise FileNotFoundError("Bottle test folder or ground-truth folder is missing.")

print("PATCHCORE GAUSSIAN SMOOTHING EXPERIMENT")
print("Dataset: MVTec AD Bottle (83 test images)")
print("This is an exploratory experiment using test masks.")
print("Do not use the best test-set result as an unbiased final score.")


# =========================================================
# 4. LOAD PATCHCORE
# =========================================================
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=6,
    coreset_sampling_ratio=0.1,
    precision="float32",
)

engine = Engine(accelerator="cpu", devices=1)


# =========================================================
# 5. COLLECT ANOMALY MAPS AND GROUND-TRUTH MASKS
# =========================================================
records = []

for category in CATEGORIES:
    category_folder = TEST_DIR / category
    image_paths = sorted(
        p for p in category_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"Skipping empty category: {category}")
        continue

    print(f"\nPredicting {category}: {len(image_paths)} images")

    dataset = PredictDataset(
        path=category_folder,
        image_size=IMAGE_SIZE,
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT),
        return_predictions=True,
    )

    if predictions is None:
        raise RuntimeError(f"No predictions returned for {category}.")

    if not isinstance(predictions, (list, tuple)):
        predictions = list(predictions)

    prediction_by_name = {
        get_prediction_path(prediction).name: prediction
        for prediction in predictions
    }

    for image_path in image_paths:
        prediction = prediction_by_name.get(image_path.name)
        if prediction is None:
            raise RuntimeError(f"Prediction not found for {image_path.name}")

        anomaly_map = to_2d_map(get_field(prediction, "anomaly_map"))

        if category == "good":
            gt_mask = np.zeros(anomaly_map.shape, dtype=np.uint8)
        else:
            mask_path = (
                GROUND_TRUTH_DIR / category / f"{image_path.stem}_mask.png"
            )
            if not mask_path.exists():
                raise FileNotFoundError(f"Ground-truth mask missing: {mask_path}")
            gt_mask = load_ground_truth_mask(mask_path, anomaly_map.shape)

        records.append({
            "image_name": image_path.name,
            "category": category,
            "anomaly_map": anomaly_map,
            "ground_truth": gt_mask,
        })

        print(
            f"{category}/{image_path.name}: "
            f"map={anomaly_map.shape}, GT pixels={int(gt_mask.sum())}"
        )


# =========================================================
# 6. COMPARE RAW AND SMOOTHED MAPS
# =========================================================
all_ground_truth = np.concatenate([
    record["ground_truth"].ravel() for record in records
])

results = []
method_maps = {}

for sigma in SIGMAS:
    method_name = "raw" if sigma == 0 else f"gaussian_sigma_{sigma:g}"
    print(f"\nAnalyzing {method_name}...")

    smoothed_maps = []
    for record in records:
        anomaly_map = record["anomaly_map"]

        if sigma == 0:
            processed_map = anomaly_map
        else:
            processed_map = gaussian_filter(
                anomaly_map,
                sigma=sigma,
                mode="reflect",
            )

        smoothed_maps.append(processed_map.astype(np.float32).ravel())

    all_scores = np.concatenate(smoothed_maps)
    method_maps[method_name] = smoothed_maps

    try:
        pixel_auc = roc_auc_score(all_ground_truth, all_scores)
    except ValueError:
        pixel_auc = float("nan")

    for percentile in PERCENTILES:
        threshold = float(np.percentile(all_scores, percentile))
        metrics = calculate_metrics(
            all_ground_truth,
            all_scores,
            threshold,
        )

        results.append({
            "method": method_name,
            "sigma": sigma,
            "percentile": percentile,
            "threshold": threshold,
            "pixel_AUROC": pixel_auc,
            **metrics,
        })


results_df = pd.DataFrame(results)
results_csv = OUTPUT_DIR / "smoothing_threshold_results.csv"
results_df.to_csv(results_csv, index=False)


# =========================================================
# 7. SUMMARIZE EACH SMOOTHING METHOD
# =========================================================
summary_rows = []

for method_name, group in results_df.groupby("method", sort=False):
    best = group.loc[group["f1"].idxmax()]
    summary_rows.append({
        "method": method_name,
        "sigma": best["sigma"],
        "best_percentile": best["percentile"],
        "threshold": best["threshold"],
        "pixel_AUROC": best["pixel_AUROC"],
        "precision": best["precision"],
        "recall": best["recall"],
        "best_pixel_F1": best["f1"],
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = OUTPUT_DIR / "smoothing_method_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print("\n==============================================")
print("BEST EXPLORATORY RESULT FOR EACH METHOD")
print("==============================================")
print(summary_df.to_string(index=False))
print(
    "\nCaution: each best F1 is selected using the test masks. "
    "These are exploratory comparisons, not unbiased final test scores."
)


# =========================================================
# 8. PLOT BEST F1 BY SMOOTHING METHOD
# =========================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(summary_df["method"], summary_df["best_pixel_F1"])
ax.set_xlabel("Anomaly-map processing method")
ax.set_ylabel("Best exploratory pixel F1-score")
ax.set_ylim(0, 1)
ax.set_title("Pixel F1-score: Raw vs Gaussian-smoothed anomaly maps")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

f1_plot = OUTPUT_DIR / "smoothing_f1_comparison.png"
fig.savefig(f1_plot, dpi=200, bbox_inches="tight")
plt.close(fig)


# =========================================================
# 9. PLOT PIXEL AUROC BY SMOOTHING METHOD
# =========================================================
auc_summary = summary_df[["method", "pixel_AUROC"]].drop_duplicates()

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(auc_summary["method"], auc_summary["pixel_AUROC"])
ax.set_xlabel("Anomaly-map processing method")
ax.set_ylabel("Pixel AUROC")
ax.set_ylim(0, 1.05)
ax.set_title("Pixel AUROC: Raw vs Gaussian-smoothed anomaly maps")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

auc_plot = OUTPUT_DIR / "smoothing_auc_comparison.png"
fig.savefig(auc_plot, dpi=200, bbox_inches="tight")
plt.close(fig)


# =========================================================
# 10. CREATE VISUAL EXAMPLES FOR A CONTAMINATION IMAGE
# =========================================================
contamination_records = [
    record for record in records
    if record["category"] == "contamination"
]

if contamination_records:
    example = contamination_records[0]
    example_name = example["image_name"]
    example_map = example["anomaly_map"]
    example_gt = example["ground_truth"]

    # Select each method's best exploratory threshold.
    selected_methods = []
    for _, row in summary_df.iterrows():
        method_name = row["method"]
        selected_methods.append((
            method_name,
            float(row["threshold"]),
        ))

    columns = 3
    rows_count = int(np.ceil((2 + len(selected_methods)) / columns))
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(12, rows_count * 4),
    )
    axes = np.asarray(axes).reshape(-1)

    axes[0].imshow(example_gt, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground-truth mask")
    axes[0].axis("off")

    axes[1].imshow(example_map, cmap="jet")
    axes[1].set_title("Raw anomaly map")
    axes[1].axis("off")

    for index, (method_name, threshold) in enumerate(
        selected_methods, start=2
    ):
        sigma = 0.0 if method_name == "raw" else float(
            method_name.replace("gaussian_sigma_", "")
        )

        processed = (
            example_map if sigma == 0
            else gaussian_filter(example_map, sigma=sigma, mode="reflect")
        )

        mask = (processed >= threshold).astype(np.uint8)
        axes[index].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[index].set_title(
            f"{method_name}\nthreshold={threshold:.4f}"
        )
        axes[index].axis("off")

    for index in range(2 + len(selected_methods), len(axes)):
        axes[index].axis("off")

    fig.suptitle(
        f"Contamination example: {example_name}\n"
        "Each method uses its best exploratory threshold",
        fontsize=13,
    )
    fig.tight_layout()

    example_path = OUTPUT_DIR / "smoothing_mask_example.png"
    fig.savefig(example_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Example mask comparison:", example_path)


# =========================================================
# 11. FINAL OUTPUTS
# =========================================================
print("\n==============================================")
print("SMOOTHING ANALYSIS COMPLETED")
print("==============================================")
print("Detailed threshold results:", results_csv)
print("Method summary:", summary_csv)
print("F1 comparison chart:", f1_plot)
print("AUROC comparison chart:", auc_plot)
print("Output folder:", OUTPUT_DIR)
