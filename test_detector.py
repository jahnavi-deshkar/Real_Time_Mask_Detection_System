import cv2
import glob
import os
from src.detector import PPEDetector

# Load the detector once
detector = PPEDetector("models/best.onnx")

# Find all JPG and PNG images in data folder
image_files = (
    glob.glob("data/*.jpg") +
    glob.glob("data/*.jpeg") +
    glob.glob("data/*.png")
)

if not image_files:
    print("No images found in the data folder.")
    exit()

# Create a folder for results
os.makedirs("data/results", exist_ok=True)

print(f"Found {len(image_files)} images.\n")

total_violations = 0

for image_path in image_files:

    # Read image
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Could not read: {image_path}")
        continue

    # Run detection
    result, violation_count = detector.process_frame(frame)

    # Create output filename
    filename = os.path.basename(image_path)
    output_path = os.path.join("data/results", filename)

    # Save result
    cv2.imwrite(output_path, result)

    total_violations += violation_count

    print(f"{filename}")
    print(f"  Violations detected: {violation_count}")
    print(f"  Result: {output_path}")
    print()

print("--------------------------------")
print(f"Total images: {len(image_files)}")
print(f"Total violations: {total_violations}")
print("--------------------------------")