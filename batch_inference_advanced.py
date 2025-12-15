import sys
from ultralytics import YOLO
from pathlib import Path
import shutil

if len(sys.argv) < 2:
    print("Usage: python batch_inference_advanced.py <model_folder_name>")
    sys.exit(1)

model_folder = sys.argv[1]

# Try different possible model paths
possible_paths = [
    Path(model_folder) / 'weights' / 'best.pt',
    Path(model_folder) / 'best.pt'
]

model_path = None
for path in possible_paths:
    if path.exists():
        model_path = path
        break

if model_path is None:
    print(f"Error: Model not found in {model_folder}")
    print(f"Tried: {[str(p) for p in possible_paths]}")
    sys.exit(1)

# Load model
print(f"Loading model: {model_path}")
model = YOLO(str(model_path))

# Test images
test_images_dir = Path('testImage')
if not test_images_dir.exists():
    print(f"Error: Test images directory not found: {test_images_dir}")
    sys.exit(1)

# Create results directory
results_dir = Path(model_folder) / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

# Clear previous results
for old_result in results_dir.glob('*'):
    if old_result.is_file():
        old_result.unlink()

print(f"\nRunning inference on test images...")
print(f"Results will be saved to: {results_dir}\n")

# Run inference
test_images = sorted(test_images_dir.glob('*.png'))

for img_path in test_images:
    print(f"Processing: {img_path.name}")

    results = model.predict(
        source=str(img_path),
        save=True,
        project=str(results_dir.parent),
        name='results',
        exist_ok=True,
        conf=0.25,
        iou=0.7
    )

print(f"\nInference complete!")
print(f"Results saved to: {results_dir}")
print(f"Total images processed: {len(test_images)}")
