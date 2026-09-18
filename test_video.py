import cv2
from src.detector import PPEDetector

detector = PPEDetector("models/best.onnx")

video_path = "data/sample_video.mp4"
output_path = "data/results_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

frame_count = 0
total_violations = 0

print("Processing video...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    result, violation_count = detector.process_frame(frame)

    out.write(result)

    total_violations += violation_count

    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()

print()
print("--------------------------------")
print("Video processing complete!")
print(f"Frames processed: {frame_count}")
print(f"Total detections: {total_violations}")
print(f"Output saved to: {output_path}")
print("--------------------------------")