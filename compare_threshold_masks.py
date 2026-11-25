from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# =========================================================
# SETTINGS
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_DIR / "datasets" / "MVTecAD" / "bottle"
TEST_DIR = DATASET_ROOT / "test"
GROUND_TRUTH_DIR = DATASET_ROOT / "ground_truth"
CHECKPOINT = (
    PROJECT_DIR / "results" / "Patchcore" / "MVTecAD" / "bottle"
    / "v0" / "weights" / "lightning" / "model.ckpt"
)

OUTPUT_DIR = PROJECT_DIR / "detailed_evaluation" / "threshold_mask_comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY = "contamination"
THRESHOLD = 0.57744930
MAX_IMAGES = 6
IMAGE_SIZE = (256, 256)


# =========================================================
# HELPERS
# =========================================================
def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def get_field(prediction, name):
    if isinstance(prediction, dict):
        return prediction[name]
    return getattr(prediction, name)


def get_image_path(prediction):
    value = get_field(prediction, "image_path")
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return Path(str(value))


def squeeze_map(value):
    array = to_numpy(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D map, got shape {array.shape}")
    return array


def resize_rgb(image_path, shape):
    height, width = shape
    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image)


def resize_binary_mask(mask_path, shape):
    height, width = shape
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def resize_prediction_mask(mask, shape):
    mask = squeeze_map(mask)
    height, width = shape
    if mask.shape != (height, width):
        mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)
        mask = (np.asarray(mask_img) > 0).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)
    return mask


# =========================================================
# CHECK INPUTS
# =========================================================
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

image_folder = TEST_DIR / CATEGORY
if not image_folder.exists():
    raise FileNotFoundError(f"Test folder not found: {image_folder}")

mask_folder = GROUND_TRUTH_DIR / CATEGORY
if not mask_folder.exists():
    raise FileNotFoundError(f"Ground-truth folder not found: {mask_folder}")


# =========================================================
# LOAD MODEL AND PREDICT
# =========================================================
print("Loading PatchCore checkpoint...")
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=6,
    coreset_sampling_ratio=0.1,
    precision="float32",
)
engine = Engine(accelerator="cpu", devices=1)

dataset = PredictDataset(path=image_folder, image_size=IMAGE_SIZE)
predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=str(CHECKPOINT),
    return_predictions=True,
)

if predictions is None:
    raise RuntimeError("No predictions were returned.")
if not isinstance(predictions, (list, tuple)):
    predictions = list(predictions)

prediction_by_name = {
    get_image_path(prediction).name: prediction
    for prediction in predictions
}

image_paths = sorted(
    p for p in image_folder.iterdir()
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
)[:MAX_IMAGES]

if not image_paths:
    raise RuntimeError("No contamination test images were found.")

# Keep individual figures and also assemble them into a contact sheet.
contact_sheet_images = []

for image_path in image_paths:
    prediction = prediction_by_name.get(image_path.name)
    if prediction is None:
        print(f"Skipping {image_path.name}: prediction not matched.")
        continue

    anomaly_map = squeeze_map(get_field(prediction, "anomaly_map")).astype(np.float32)
    target_shape = anomaly_map.shape

    gt_path = mask_folder / f"{image_path.stem}_mask.png"
    if not gt_path.exists():
        print(f"Skipping {image_path.name}: ground-truth mask missing.")
        continue

    original = resize_rgb(image_path, target_shape)
    ground_truth = resize_binary_mask(gt_path, target_shape)

    # The standard Anomalib post-processed mask from the model output.
    current_mask = resize_prediction_mask(
        get_field(prediction, "pred_mask"),
        target_shape,
    )

    # Experimental mask created directly from the raw anomaly map.
    experimental_mask = (anomaly_map >= THRESHOLD).astype(np.uint8)

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    panels = [
        (original, "Original", None),
        (ground_truth, "Ground-truth mask", "gray"),
        (current_mask, "Current Anomalib mask", "gray"),
        (experimental_mask, f"Threshold mask\n({THRESHOLD:.4f})", "gray"),
        (anomaly_map, "Anomaly map", "jet"),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=cmap, vmin=0 if cmap == "gray" else None,
                      vmax=1 if cmap == "gray" else None)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"Contamination localization: {image_path.name}")
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{image_path.stem}_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Create a compact thumbnail for the contact sheet.
    contact_sheet_images.append((image_path.name, output_path))
    print(f"Saved: {output_path}")

# Build a contact sheet from the saved comparison images.
if contact_sheet_images:
    columns = 2
    rows = math.ceil(len(contact_sheet_images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(18, rows * 4.8))
    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (name, path) in zip(axes, contact_sheet_images):
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(
        f"Contamination: current mask vs threshold {THRESHOLD:.4f}",
        fontsize=15,
    )
    fig.tight_layout()
    sheet_path = OUTPUT_DIR / "contamination_threshold_contact_sheet.png"
    fig.savefig(sheet_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nContact sheet saved:", sheet_path)

print("\nComparison completed.")
print("Output folder:", OUTPUT_DIR)
print("Note: threshold 0.57744930 was selected using the test set for exploration.")
print("Do not treat it as a validated production threshold.")
