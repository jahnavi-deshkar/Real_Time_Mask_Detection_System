import cv2
import numpy as np
import onnxruntime as ort


class PPEDetector:

    def __init__(self, model_path="models/best.onnx"):

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        # Class mapping verified from our dataset
        # 0 = No Mask
        # 1 = Mask
        self.class_names = {
            0: "No Mask",
            1: "Mask"
        }

    def preprocess(self, frame):

        original_height, original_width = frame.shape[:2]

        image = cv2.resize(frame, (640, 640))

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize
        image = image.astype(np.float32) / 255.0

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image, original_width, original_height

    def process_frame(self, frame, confidence_threshold=0.40):

        input_tensor, original_width, original_height = self.preprocess(frame)

        outputs = self.session.run(
            None,
            {self.input_name: input_tensor}
        )

        predictions = outputs[0][0]

        violation_count = 0

        x_scale = original_width / 640
        y_scale = original_height / 640

        for detection in predictions:

            x1, y1, x2, y2, confidence, class_id = detection

            confidence = float(confidence)
            class_id = int(class_id)

            if confidence < confidence_threshold:
                continue

            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            x1 = max(0, min(x1, original_width - 1))
            y1 = max(0, min(y1, original_height - 1))
            x2 = max(0, min(x2, original_width - 1))
            y2 = max(0, min(y2, original_height - 1))

            class_name = self.class_names.get(
                class_id,
                f"Class {class_id}"
            )

            if class_id == 0:
                # No Mask = violation
                violation_count += 1
                box_color = (0, 0, 255)  # Red
            else:
                # Mask = compliant
                box_color = (0, 255, 0)  # Green

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            label = f"{class_name}: {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

        return frame, violation_count