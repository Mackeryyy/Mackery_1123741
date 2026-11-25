"""
PatchCore training-size experiment.

Trains four independent PatchCore models using 25%, 50%, 75%, and 100%
of the normal MVTec AD Bottle training images, then evaluates every model
on the SAME fixed 83-image test set.

Run from the project root:
    python training_size_experiment.py

Outputs:
    training_size_experiments/training_size_results.csv
    training_size_experiments/image_auroc_vs_training_size.png
    training_size_experiments/pixel_auroc_vs_training_size.png
    training_size_experiments/experiment_summary.txt

Important:
- This is an experimental comparison, not an epoch-based learning curve.
- The script does not modify your original dataset or existing checkpoint.
- The four models are saved in separate experiment folders.
"""

from pathlib import Path
import random
import shutil
import csv
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score

import torch
from anomalib.data import MVTecAD, PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ---------------------------------------------------------
# 1. PATHS AND EXPERIMENT SETTINGS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_CATEGORY_DIR = PROJECT_DIR / "datasets" / "MVTecAD" / "bottle"
SOURCE_TRAIN_DIR = SOURCE_CATEGORY_DIR / "train" / "good"
SOURCE_TEST_DIR = SOURCE_CATEGORY_DIR / "test"
SOURCE_MASK_DIR = SOURCE_CATEGORY_DIR / "ground_truth"

OUTPUT_DIR = PROJECT_DIR / "training_size_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
RANDOM_SEED = 42
IMAGE_SIZE = (256, 256)
CATEGORIES = ["good", "broken_large", "broken_small", "contamination"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# Same PatchCore settings as the supplied train.py.
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
NUM_NEIGHBORS = 6
CORESET_RATIO = 0.1
PRECISION = "float32"


# ---------------------------------------------------------
# 2. SMALL HELPER FUNCTIONS
# ---------------------------------------------------------

def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value):
    return float(to_numpy(value).reshape(-1)[0])


def safe_auc(labels, scores):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)

    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")

    return float(roc_auc_score(labels, scores))


def list_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_experiment_dataset(fraction, selected_train_images):
    """
    Create an isolated MVTec AD Bottle directory for one training fraction.
    Test images and ground-truth masks are copied unchanged.
    Only train/good is reduced.
    """
    percentage = int(round(fraction * 100))
    experiment_root = OUTPUT_DIR / f"data_{percentage}" / "MVTecAD"
    category_root = experiment_root / "bottle"

    if category_root.exists():
        shutil.rmtree(category_root)

    (category_root / "train" / "good").mkdir(parents=True, exist_ok=True)

    for image_path in selected_train_images:
        shutil.copy2(
            image_path,
            category_root / "train" / "good" / image_path.name
        )

    # Copy the same fixed test set for every experiment.
    shutil.copytree(
        SOURCE_TEST_DIR,
        category_root / "test",
        dirs_exist_ok=True
    )

    # Copy the same ground-truth masks for every experiment.
    shutil.copytree(
        SOURCE_MASK_DIR,
        category_root / "ground_truth",
        dirs_exist_ok=True
    )

    return experiment_root


def create_model():
    return Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        pre_trained=True,
        num_neighbors=NUM_NEIGHBORS,
        coreset_sampling_ratio=CORESET_RATIO,
        precision=PRECISION,
    )


def evaluate_checkpoint(checkpoint_path):
    """
    Evaluate image AUROC and pixel AUROC using the same 83 test images.
    Pixel masks are compared with the MVTec AD ground-truth masks.
    """
    model = create_model()
    engine = Engine(accelerator="cpu", devices=1)

    image_labels = []
    image_scores = []

    all_gt_pixels = []
    all_anomaly_pixels = []

    category_image_counts = {}

    for category in CATEGORIES:
        category_dir = SOURCE_TEST_DIR / category
        image_paths = list_images(category_dir)

        if not image_paths:
            continue

        dataset = PredictDataset(
            path=category_dir,
            image_size=IMAGE_SIZE
        )

        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=str(checkpoint_path),
            return_predictions=True,
        )

        if predictions is None:
            raise RuntimeError(
                f"No predictions returned for category: {category}"
            )

        if not isinstance(predictions, (list, tuple)):
            predictions = list(predictions)

        prediction_by_name = {}
        for prediction in predictions:
            image_path_value = getattr(prediction, "image_path", None)
            if image_path_value is None and isinstance(prediction, dict):
                image_path_value = prediction.get("image_path")

            if isinstance(image_path_value, (list, tuple, np.ndarray)):
                image_path_value = image_path_value[0]

            if hasattr(image_path_value, "item"):
                image_path_value = image_path_value.item()

            if image_path_value is not None:
                prediction_by_name[Path(str(image_path_value)).name] = prediction

        category_image_counts[category] = len(image_paths)

        for image_path in image_paths:
            prediction = prediction_by_name.get(image_path.name)
            if prediction is None:
                raise RuntimeError(
                    f"Could not match prediction for {image_path.name}"
                )

            if isinstance(prediction, dict):
                anomaly_map_value = prediction["anomaly_map"]
                score_value = prediction["pred_score"]
                mask_value = prediction.get("pred_mask")
            else:
                anomaly_map_value = prediction.anomaly_map
                score_value = prediction.pred_score
                mask_value = getattr(prediction, "pred_mask", None)

            anomaly_map = np.squeeze(
                to_numpy(anomaly_map_value)
            ).astype(np.float32)

            image_score = scalar(score_value)

            # MVTec AD label convention used in the project's evaluation:
            # good = 0, defective categories = 1.
            image_labels.append(0 if category == "good" else 1)
            image_scores.append(image_score)

            if mask_value is None:
                raise RuntimeError(
                    "The model prediction did not contain pred_mask. "
                    "Pixel AUROC cannot be computed with this script."
                )

            pred_mask = np.squeeze(to_numpy(mask_value))
            if pred_mask.ndim != 2:
                raise ValueError(
                    f"Unexpected predicted mask shape: {pred_mask.shape}"
                )

            if category == "good":
                gt_mask = np.zeros(anomaly_map.shape, dtype=np.uint8)
            else:
                gt_path = (
                    SOURCE_MASK_DIR
                    / category
                    / f"{image_path.stem}_mask.png"
                )
                if not gt_path.exists():
                    raise FileNotFoundError(
                        f"Ground-truth mask not found: {gt_path}"
                    )

                gt_image = Image.open(gt_path).convert("L")
                gt_image = gt_image.resize(
                    (anomaly_map.shape[1], anomaly_map.shape[0]),
                    Image.Resampling.NEAREST
                )
                gt_mask = (
                    np.asarray(gt_image) > 0
                ).astype(np.uint8)

            # Match mask dimensions before combining pixel scores.
            if pred_mask.shape != gt_mask.shape:
                pred_image = Image.fromarray(
                    (pred_mask > 0).astype(np.uint8) * 255
                )
                pred_image = pred_image.resize(
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    Image.Resampling.NEAREST
                )
                pred_mask = (
                    np.asarray(pred_image) > 0
                ).astype(np.uint8)

            all_gt_pixels.append(gt_mask.reshape(-1))
            all_anomaly_pixels.append(anomaly_map.reshape(-1))

    image_auroc = safe_auc(image_labels, image_scores)
    pixel_auroc = safe_auc(
        np.concatenate(all_gt_pixels),
        np.concatenate(all_anomaly_pixels)
    )

    return {
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "test_images": len(image_labels),
        "category_counts": category_image_counts,
    }


# ---------------------------------------------------------
# 3. VALIDATE DATASET
# ---------------------------------------------------------

if not SOURCE_TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder not found: {SOURCE_TRAIN_DIR}"
    )

if not SOURCE_TEST_DIR.exists() or not SOURCE_MASK_DIR.exists():
    raise FileNotFoundError(
        "Bottle test folder or ground_truth folder was not found."
    )

normal_images = list_images(SOURCE_TRAIN_DIR)

if len(normal_images) == 0:
    raise RuntimeError("No normal training images were found.")

for category in CATEGORIES:
    if not (SOURCE_TEST_DIR / category).exists():
        raise FileNotFoundError(
            f"Test category folder is missing: {category}"
        )

random_generator = random.Random(RANDOM_SEED)
shuffled_images = normal_images.copy()
random_generator.shuffle(shuffled_images)

print("=" * 65)
print("PATCHCORE TRAINING-DATA-SIZE EXPERIMENT")
print("=" * 65)
print(f"Normal training images available: {len(normal_images)}")
print("Fixed test categories:", ", ".join(CATEGORIES))
print("Training fractions:", TRAINING_FRACTIONS)
print("CPU execution enabled.")
print()


# ---------------------------------------------------------
# 4. TRAIN AND EVALUATE EACH TRAINING FRACTION
# ---------------------------------------------------------

results = []

for fraction in TRAINING_FRACTIONS:
    percentage = int(round(fraction * 100))
    selected_count = max(1, int(round(len(normal_images) * fraction)))
    selected_images = shuffled_images[:selected_count]

    # Reset random seeds so each run starts from a reproducible state.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("\n" + "-" * 65)
    print(
        f"Experiment: {percentage}% "
        f"({selected_count}/{len(normal_images)} normal images)"
    )
    print("-" * 65)

    experiment_root = make_experiment_dataset(
        fraction,
        selected_images
    )

    datamodule = MVTecAD(
        root=str(experiment_root),
        category="bottle",
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
    )

    model = create_model()

    experiment_output = OUTPUT_DIR / f"model_{percentage}"
    engine = Engine(
        default_root_dir=str(experiment_output),
        accelerator="cpu",
        devices=1,
        max_epochs=1,
    )

    print("Starting PatchCore fitting...")
    engine.fit(model=model, datamodule=datamodule)

    # Find the checkpoint produced by this isolated experiment.
    checkpoint_candidates = list(
        experiment_output.rglob("*.ckpt")
    )

    if not checkpoint_candidates:
        raise FileNotFoundError(
            f"No checkpoint was created for {percentage}% experiment "
            f"under {experiment_output}"
        )

    # Prefer the latest checkpoint if more than one exists.
    checkpoint_path = max(
        checkpoint_candidates,
        key=lambda path: path.stat().st_mtime
    )

    print(f"Checkpoint: {checkpoint_path}")
    print("Evaluating against the fixed Bottle test set...")

    metrics = evaluate_checkpoint(checkpoint_path)

    row = {
        "training_fraction_percent": percentage,
        "normal_training_images": selected_count,
        "total_available_normal_images": len(normal_images),
        "test_images": metrics["test_images"],
        "image_AUROC": metrics["image_auroc"],
        "pixel_AUROC": metrics["pixel_auroc"],
        "checkpoint": str(checkpoint_path),
    }
    results.append(row)

    print(
        f"Image AUROC: {metrics['image_auroc']:.6f}"
        if np.isfinite(metrics["image_auroc"])
        else "Image AUROC: undefined"
    )
    print(
        f"Pixel AUROC: {metrics['pixel_auroc']:.6f}"
        if np.isfinite(metrics["pixel_auroc"])
        else "Pixel AUROC: undefined"
    )

    # Release resources before the next CPU experiment.
    del engine, model, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------
# 5. SAVE RESULTS AND PLOTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)
csv_path = OUTPUT_DIR / "training_size_results.csv"
results_df.to_csv(csv_path, index=False)

def plot_metric(column, title, y_label, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        results_df["training_fraction_percent"],
        results_df[column],
        marker="o",
        linewidth=2,
    )
    ax.set_title(title)
    ax.set_xlabel("Normal Training Images Used (%)")
    ax.set_ylabel(y_label)
    ax.set_xticks([
        int(round(value * 100)) for value in TRAINING_FRACTIONS
    ])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close(fig)


plot_metric(
    "image_AUROC",
    "Image AUROC vs. Training Dataset Size",
    "Image AUROC",
    "image_auroc_vs_training_size.png",
)

plot_metric(
    "pixel_AUROC",
    "Pixel AUROC vs. Training Dataset Size",
    "Pixel AUROC",
    "pixel_auroc_vs_training_size.png",
)

summary_path = OUTPUT_DIR / "experiment_summary.txt"
with open(summary_path, "w", encoding="utf-8") as file:
    file.write("PATCHCORE TRAINING-DATA-SIZE EXPERIMENT\n")
    file.write("All experiments use the same fixed test dataset.\n")
    file.write("The training subset is sampled from train/good only.\n")
    file.write("Random seed: 42\n\n")
    file.write(results_df.to_string(index=False))
    file.write("\n")

print("\n" + "=" * 65)
print("EXPERIMENTS COMPLETED")
print("=" * 65)
print(results_df.to_string(index=False))
print("\nSaved results:")
print(csv_path)
print(OUTPUT_DIR / "image_auroc_vs_training_size.png")
print(OUTPUT_DIR / "pixel_auroc_vs_training_size.png")
print(summary_path)
