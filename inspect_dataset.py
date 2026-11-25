
from pathlib import Path

dataset = Path("datasets/MVTecAD/bottle")

folders = [
    dataset / "train" / "good",
    dataset / "test" / "good",
    dataset / "test" / "broken_large",
    dataset / "test" / "broken_small",
    dataset / "test" / "contamination",
]

print("MVTec AD - Bottle Dataset Inspection")
print("=" * 45)

for folder in folders:
    images = list(folder.glob("*.png"))
    print(f"{folder.name}: {len(images)} images")

print("\nGround Truth Masks")
print("=" * 45)

for folder in (dataset / "ground_truth").iterdir():
    masks = list(folder.glob("*.png"))
    print(f"{folder.name}: {len(masks)} masks")
