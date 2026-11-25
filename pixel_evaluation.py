from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score

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


def get_prediction_field(prediction, field):
    if isinstance(prediction, dict):
        return prediction[field]
    return getattr(prediction, field)


def resize_mask(mask_path, target_shape):
    height, width = target_shape
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def safe_auc(y_true, scores):
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def make_comparison_figure(original_path, gt_mask, pred_mask, anomaly_map, out_path, title):
    original = Image.open(original_path).convert("RGB")
    h, w = gt_mask.shape
    original = original.resize((w, h), Image.Resampling.BILINEAR)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-truth mask")
    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted mask")
    heat = axes[3].imshow(anomaly_map, cmap="jet")
    axes[3].set_title("Anomaly map")
    fig.colorbar(heat, ax=axes[3], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# 2. CHECK FILES AND LOAD PATCHCORE
# =====================================================
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
if not TEST_DIR.exists() or not MASK_ROOT.exists():
    raise FileNotFoundError("MVTec AD bottle test or ground_truth folder not found.")

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=6,
    coreset_sampling_ratio=0.1,
    precision="float32",
)
engine = Engine(accelerator="cpu", devices=1)

records = []
all_gt_pixels = []
all_score_pixels = []
all_pred_pixels = []
category_pixel_data = {}
representative_saved = set()

print("Starting pixel-level evaluation of MVTec AD Bottle...")

# =====================================================
# 3. RUN PREDICTIONS AND COMPARE WITH GROUND-TRUTH MASKS
# =====================================================
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

    # Match predictions to source images by their image_path.
    prediction_by_name = {}
    for prediction in predictions:
        pred_path = get_prediction_path(prediction)
        if pred_path is not None:
            prediction_by_name[pred_path.name] = prediction

    category_gt = []
    category_scores = []
    category_pred = []

    for image_path in image_paths:
        prediction = prediction_by_name.get(image_path.name)
        if prediction is None:
            raise RuntimeError(
                f"Could not match prediction for {image_path.name} in {category}."
            )

        anomaly_map = np.squeeze(as_numpy(get_prediction_field(prediction, "anomaly_map"))).astype(np.float32)
        pred_mask_value = get_prediction_field(prediction, "pred_mask")
        if pred_mask_value is None:
            raise RuntimeError(
                "Anomalib did not return pred_mask. Cannot calculate pixel F1 "
                "using the model's predicted segmentation mask."
            )
        pred_mask = np.squeeze(as_numpy(pred_mask_value))
        if pred_mask.ndim != 2:
            raise ValueError(f"Unexpected predicted mask shape: {pred_mask.shape}")
        pred_mask = (pred_mask > 0).astype(np.uint8)

        if category == "good":
            gt_mask = np.zeros(anomaly_map.shape, dtype=np.uint8)
        else:
            mask_path = MASK_ROOT / category / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")
            gt_mask = resize_mask(mask_path, anomaly_map.shape)

        # Ensure prediction and ground truth use identical dimensions.
        if pred_mask.shape != gt_mask.shape:
            pred_mask_img = Image.fromarray(pred_mask * 255)
            pred_mask_img = pred_mask_img.resize(
                (gt_mask.shape[1], gt_mask.shape[0]), Image.Resampling.NEAREST
            )
            pred_mask = (np.asarray(pred_mask_img) > 0).astype(np.uint8)

        gt_flat = gt_mask.reshape(-1)
        score_flat = anomaly_map.reshape(-1)
        pred_flat = pred_mask.reshape(-1)

        category_gt.append(gt_flat)
        category_scores.append(score_flat)
        category_pred.append(pred_flat)
        all_gt_pixels.append(gt_flat)
        all_score_pixels.append(score_flat)
        all_pred_pixels.append(pred_flat)

        records.append({
            "image_name": image_path.name,
            "defect_type": category,
            "ground_truth_defect_pixels": int(gt_flat.sum()),
            "predicted_defect_pixels": int(pred_flat.sum()),
            "pixel_f1": float(f1_score(gt_flat, pred_flat, zero_division=0)),
            "pixel_auroc": safe_auc(gt_flat, score_flat),
        })

        # Save one visual example for each defect category.
        if category != "good" and category not in representative_saved:
            comparison_path = OUTPUT_DIR / f"pixel_comparison_{category}.png"
            make_comparison_figure(
                image_path, gt_mask, pred_mask, anomaly_map,
                comparison_path, f"Pixel Localization - {category}"
            )
            representative_saved.add(category)

        print(
            f"{category}/{image_path.name}: "
            f"GT defect pixels={int(gt_flat.sum())}, "
            f"predicted defect pixels={int(pred_flat.sum())}"
        )

    category_gt_flat = np.concatenate(category_gt)
    category_scores_flat = np.concatenate(category_scores)
    category_pred_flat = np.concatenate(category_pred)
    category_pixel_data[category] = {
        "images": len(image_paths),
        "pixel_auroc": safe_auc(category_gt_flat, category_scores_flat),
        "pixel_f1": float(f1_score(category_gt_flat, category_pred_flat, zero_division=0)),
        "ground_truth_pixels": int(category_gt_flat.sum()),
        "predicted_pixels": int(category_pred_flat.sum()),
    }

# =====================================================
# 4. SAVE METRICS
# =====================================================
all_gt = np.concatenate(all_gt_pixels)
all_scores = np.concatenate(all_score_pixels)
all_pred = np.concatenate(all_pred_pixels)

overall = {
    "scope": "All 83 bottle test images",
    "total_images": len(records),
    "pixel_AUROC": safe_auc(all_gt, all_scores),
    "pixel_F1Score": float(f1_score(all_gt, all_pred, zero_division=0)),
    "ground_truth_defect_pixels": int(all_gt.sum()),
    "predicted_defect_pixels": int(all_pred.sum()),
}

summary_rows = [{"Category": "ALL", **overall}]
for category, metrics in category_pixel_data.items():
    summary_rows.append({
        "Category": category,
        "scope": f"{metrics['images']} images in {category}",
        "total_images": metrics["images"],
        "pixel_AUROC": metrics["pixel_auroc"],
        "pixel_F1Score": metrics["pixel_f1"],
        "ground_truth_defect_pixels": metrics["ground_truth_pixels"],
        "predicted_defect_pixels": metrics["predicted_pixels"],
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / "pixel_level_summary.csv"
summary_df.to_csv(summary_path, index=False)

per_image_df = pd.DataFrame(records)
per_image_path = OUTPUT_DIR / "pixel_level_per_image.csv"
per_image_df.to_csv(per_image_path, index=False)

print("\n========================================")
print("PIXEL-LEVEL EVALUATION RESULTS")
print("========================================")
print(summary_df.to_string(index=False))
print("\nSaved:")
print(summary_path)
print(per_image_path)
print("Comparison images:")
for category in sorted(representative_saved):
    print(OUTPUT_DIR / f"pixel_comparison_{category}.png")
print("\nPixel-level evaluation completed.")
