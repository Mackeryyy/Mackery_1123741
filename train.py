
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

print("Starting PatchCore training...")

# 1. Load the Bottle dataset
datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="bottle",
    train_batch_size=4,
    eval_batch_size=4,
    num_workers=0,
)

# 2. Create the PatchCore model
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=6,
    coreset_sampling_ratio=0.1,
    precision="float32",
)

# 3. Configure the training engine
engine = Engine(
    default_root_dir="./results",
    accelerator="cpu",
    devices=1,
    max_epochs=1,
)

# 4. Train using normal images
engine.fit(
    model=model,
    datamodule=datamodule,
)

print("Training completed!")

# 5. Evaluate on the test dataset
results = engine.test(
    model=model,
    datamodule=datamodule,
)

print("Evaluation results:")
print(results)
