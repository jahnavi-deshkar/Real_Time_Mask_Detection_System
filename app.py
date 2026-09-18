import streamlit as st
import cv2
import tempfile
import os

from src.detector import PPEDetector
from src.tracker import ViolationTracker


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Mask Detection",
    page_icon="😷",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("😷 PPE / Mask Detection System")

st.write(
    "Upload a video to detect people wearing masks "
    "and people without masks."
)


# -----------------------------
# Load detector
# -----------------------------

@st.cache_resource
def load_detector():

    return PPEDetector("models/best.onnx")


detector = load_detector()


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)


# -----------------------------
# Upload video
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)


if uploaded_file is not None:

    # Save uploaded video temporarily

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_file.write(uploaded_file.read())
    temp_file.close()

    video_path = temp_file.name

    # Open video

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        st.error("Could not open the uploaded video.")

    else:

        st.success("Video loaded successfully!")

        # Video information

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        # Display areas

        video_placeholder = st.empty()

        metric_placeholder = st.empty()

        total_violations = 0

        frame_count = 0

        # Process video

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Detect masks

            result, violation_count = detector.process_frame(
                frame,
                confidence_threshold=confidence
            )

            total_violations += violation_count

            frame_count += 1

            # Convert BGR to RGB

            result_rgb = cv2.cvtColor(
                result,
                cv2.COLOR_BGR2RGB
            )

            # Display frame

            video_placeholder.image(
                result_rgb,
                channels="RGB",
                use_container_width=True
            )

            # Display violation count

            metric_placeholder.metric(
                "Violations Detected",
                total_violations
            )

        cap.release()

        # Delete temporary file

        os.unlink(video_path)

        st.success("Video processing completed.")

else:

    st.info(
        "Please upload a video above to start detection."
    )