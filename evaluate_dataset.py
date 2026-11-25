from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_DIR / "datasets" / "MVTecAD"

CHECKPOINT_PATH = (
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


# --------------------------------------------------
# 2. Check checkpoint
# --------------------------------------------------

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT_PATH}"
    )

print("Checkpoint found.")
print("Starting PatchCore evaluation...")


# --------------------------------------------------
# 3. Load the MVTec AD bottle dataset
# --------------------------------------------------

datamodule = MVTecAD(
    root=str(DATASET_PATH),
    category="bottle",
    train_batch_size=4,
    eval_batch_size=4,
    num_workers=0,
)


# --------------------------------------------------
# 4. Load the PatchCore model
# --------------------------------------------------

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=6,
    coreset_sampling_ratio=0.1,
    precision="float32",
)


# --------------------------------------------------
# 5. Create the evaluation engine
# --------------------------------------------------

engine = Engine(
    default_root_dir=str(PROJECT_DIR / "evaluation_results"),
    accelerator="cpu",
    devices=1,
)


# --------------------------------------------------
# 6. Evaluate the saved model
# --------------------------------------------------

results = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(CHECKPOINT_PATH),
)


# --------------------------------------------------
# 7. Display evaluation results
# --------------------------------------------------

print("\n" + "=" * 50)
print("PATCHCORE EVALUATION RESULTS")
print("=" * 50)

print("Dataset: MVTec AD - Bottle")
print("Test images: 83")
print("\nMetrics:")

for result in results:
    for metric_name, metric_value in result.items():
        if hasattr(metric_value, "item"):
            metric_value = metric_value.item()

        print(f"{metric_name}: {metric_value}")

print("\nEvaluation completed!")