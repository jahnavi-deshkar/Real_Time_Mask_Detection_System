REAL TIME MASK DETECTION SYSTEM
Project Documentation and Technical Specification
YOLOv8 • ONNX Runtime • OpenCV • Streamlit

Developed as a local computer-vision prototype for mask compliance detection
 
1. PROJECT SPECIFICATION
Project Name:  Mask Detection System
Project Type: Computer Vision • Real-Time AI • Python Application • Local Processing
Primary Purpose: Detect people wearing a mask and people without a mask in images or video, visually mark the detections, count mask violations, and provide a Streamlit interface for video upload and webcam-based detection.

2. PROJECT OVERVIEW
Objective
The project is a computer-vision prototype that uses a trained object-detection model to identify mask compliance. Each processed frame is analyzed for the two classes used by the trained model: Mask and No Mask. Detected regions are displayed with bounding boxes so that the result can be understood visually.
The implementation was developed from a pre-labelled face-mask dataset, trained using Google Colab, and then exported to ONNX so inference could run locally with ONNX Runtime. The local application uses OpenCV for frame handling and Streamlit for the user interface.

3. PRIMARY GOALS
1.	Detect masks and no-mask instances in images and video.
2.	Draw a bounding box around each detected instance.
3.	Use green boxes for Mask detections and red boxes for No Mask detections.
4.	Count detected No Mask instances as violations.
5.	Process uploaded video files frame-by-frame.
6.	Support webcam-based detection.
7.	Provide a simple confidence-threshold control.
8.	Display the annotated output and a live violation counter.
9.	Record violations using a cooldown mechanism to avoid repeated log entries.

4. TARGET USERS / USE CASES
The prototype is suitable for demonstrating automated mask-compliance monitoring in controlled environments such as workplaces, laboratories, educational facilities, construction or industrial settings, and other locations where visual mask compliance is required.
This project is a prototype and should not be treated as a certified safety or regulatory compliance system without additional validation in the intended environment.

5. DEVELOPMENT PHILOSOPHY
  •	Simplicity — keep the prototype understandable and easy to demonstrate.
  •	Modularity — separate detection, logging, and application/UI responsibilities.
  •	Readability — use descriptive names and straightforward functions.
  •	Easy debugging — test the detector and tracker independently before integration.
  •	Local operation — avoid unnecessary cloud dependencies during inference.
  •	Future expansion — keep the project structure suitable for later improvements.

6. TECHNOLOGY STACK
  Component	Technology	Role
  Programming	Python 3.10+	Application and computer-vision logic
  Object Detection	YOLOv8	Training and model generation
  Model Training	Google Colab	GPU-based training
  Dataset	Roboflow Universe mask dataset	Pre-labelled training/validation data
  Local Inference	ONNX Runtime	Runs exported best.onnx locally
  Computer Vision	OpenCV	Image/video capture, resizing, drawing
  Numerical Processing	NumPy	Tensor preparation and array operations
  User Interface	Streamlit	Browser-based local dashboard
  Logging	CSV	Simple violation records

7. HIGH-LEVEL SYSTEM PIPELINE
    Input Source → image, uploaded video, or webcam.
    Frame Acquisition → OpenCV reads an image/frame.
    Preprocessing → frame is resized to 640 × 640, converted from BGR to RGB, normalized, and converted to CHW tensor format.
    Model Inference → exported YOLOv8 model is executed through ONNX Runtime.
    Detection Filtering → detections below the selected confidence threshold are ignored.
    Classification → class 0 is interpreted as No Mask and class 1 as Mask.
    Visualization → bounding boxes and confidence labels are drawn.
    Violation Counting → each accepted No Mask detection contributes to the violation count.
    Logging → the tracker can record violations with a five-second cooldown.
    Display → Streamlit presents the processed frame and violation metric.

8. DATASET AND MODEL TRAINING
  A pre-labelled mask-detection dataset was obtained from Roboflow Universe. The selected dataset contained 6,866 images and two object-detection classes. The dataset was extracted in Google Colab and its data.yaml identified two classes.
  For the completed model, the class mapping was verified as:
  •	Class 0 → No Mask
  •	Class 1 → Mask
  The YOLOv8 Nano model was trained for 50 epochs at 640 × 640 image size with a batch size of 16 using the available GPU in Google Colab. The resulting best model was saved as best.pt.
  The trained model was subsequently exported to ONNX with a 640 × 640 input, enabling local inference through ONNX Runtime. The exported model produces up to 300 post-NMS detections in the form used by the local detector.

9. MODEL VALIDATION RESULTS
  Metric	All Classes	No Mask	Mask	Interpretation
  Precision	0.935	0.943	0.927	Proportion of reported detections that were correct
  Recall	0.908	0.911	0.904	Proportion of relevant instances detected
  mAP50	0.945	0.956	0.933	Detection performance at IoU 0.50
  mAP50-95	0.709	0.707	0.712	Average performance across stricter IoU thresholds
  Validation was performed on 582 images containing 1,059 annotated instances. These metrics describe the model validation run and should not be interpreted as guaranteed real-world performance.

10. PERSON / MASK DETECTION LOGIC
  The current implementation directly processes the model's exported detections. For every accepted detection, the detector obtains the bounding-box coordinates, confidence, and class identifier.
  •	If class_id = 0, the detection is labelled No Mask, counted as a violation, and drawn with a red box.
  •	If class_id = 1, the detection is labelled Mask and drawn with a green box.
  •	The confidence value is displayed alongside the class name.
  •	Coordinates are scaled from the 640 × 640 model space back to the original frame dimensions.
  •	Coordinates are clipped to the frame boundaries.

11. CONFIDENCE THRESHOLD
  The Streamlit application exposes a confidence threshold slider. Its current range is 0.10 to 0.90, with a default value of 0.40. A detection is ignored when its confidence is below the selected threshold.
  This setting provides a practical way to trade off sensitivity and the number of lower-confidence detections during demonstrations and testing.

12. VIDEO INPUT
The application currently supports two user-facing modes:
•	Upload Video — accepts MP4, AVI, and MOV files and processes them frame-by-frame.
•	Use Webcam — opens the computer's default camera through OpenCV.
For Streamlit display, OpenCV's BGR frame is converted to RGB before being sent to the image display.

13. VIOLATION TRACKING AND LOGGING
A separate ViolationTracker class is implemented in src/tracker.py. It creates data/logs.csv when the file does not exist and writes timestamp and violation_count columns.
To prevent log spamming, the tracker uses a five-second cooldown. A new record is written only when the current call contains one or more violations and the cooldown has expired.
Current CSV structure:
Column	Meaning
timestamp	Date and time of the logged event
violation_count	Number of No Mask detections at the logging event

14. USER INTERFACE
•	Application title: PPE / Mask Detection System.
•	Short description explaining that masks and no-mask cases are detected.
•	Sidebar confidence-threshold control.
•	Detection mode selector: Upload Video or Use Webcam.
•	Live annotated frame display.
•	Live violation counter.
•	Completion or error messages for video processing and webcam access.

15. PROJECT DIRECTORY STRUCTURE
ppe_compliance_tracker/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── best.onnx
├── data/
│   ├── sample_video.mp4
│   ├── logs.csv
│   └── results/
└── src/
    ├── __init__.py
    ├── detector.py
    └── tracker.py

16. MODULE RESPONSIBILITIES
Module	Responsibility	Current Status
app.py	Streamlit UI, video upload, webcam mode, frame display, confidence control, violation metric	Implemented
src/detector.py	Load ONNX model, preprocess frames, run inference, draw boxes, count No Mask detections	Implemented
src/tracker.py	Create CSV log, record violations, enforce five-second cooldown	Implemented and tested
models/best.onnx	Exported trained YOLOv8 detection model for local inference	Implemented
data/logs.csv	Persistent CSV violation log	Supported by tracker
test_detector.py	Batch image testing and saved annotated results	Implemented
test_single_image.py	One-image-at-a-time demonstration/test workflow	Planned/optional
test_video.py	Offline video processing and annotated output generation	Implemented/tested

17. LOCAL INFERENCE ARCHITECTURE
Because the local Windows environment blocked the PyTorch DLL required by the standard Ultralytics runtime, the completed project uses the exported ONNX model with ONNX Runtime for local inference. This preserves the trained model while avoiding the blocked PyTorch dependency during local execution.
Training remains a YOLOv8 workflow; local deployment uses best.onnx rather than loading best.pt directly.

18. FUNCTIONAL REQUIREMENTS
ID	Requirement	Priority
FR-01	Load the trained ONNX detection model at application startup.	Must
FR-02	Accept an image/frame and run object detection.	Must
FR-03	Ignore detections below the configured confidence threshold.	Must
FR-04	Identify class 0 as No Mask and class 1 as Mask.	Must
FR-05	Draw red boxes for No Mask and green boxes for Mask.	Must
FR-06	Return the annotated frame and violation count.	Must
FR-07	Process uploaded video frame-by-frame.	Must
FR-08	Provide webcam detection mode.	Must
FR-09	Display a live violation counter.	Must
FR-10	Log violations with a cooldown mechanism.	Should

19. NON-FUNCTIONAL REQUIREMENTS
•	Usability: interface should be understandable without command-line interaction after startup.
•	Modularity: detector and tracker remain separate components.
•	Maintainability: configuration and dependencies should be documented.
•	Portability: local inference should work on a standard Windows machine where the required packages are permitted.
•	Performance: processing should be responsive enough for demonstration; actual FPS depends on hardware, frame size, and model inference time.
•	Reliability: invalid video input or unavailable webcam should produce a clear error instead of silently failing.

20. TEST PLAN
Test Case	Action	Expected Result
Single image — Mask	Mask instance is detected and labelled Mask.	Green bounding box; no No Mask violation for that detection.
Single image — No Mask	No Mask instance is detected.	Red bounding box and violation count increases.
Mixed image	Both classes appear.	Correct classes receive corresponding colours and labels.
Confidence threshold	Threshold is increased/decreased.	Low-confidence detections are filtered according to the selected value.
Video processing	Local test video is processed frame-by-frame.	Annotated output video is created successfully.
Webcam	Default camera is available.	Live frames are processed and displayed.
Invalid video	Video cannot be opened.	Application displays an error.
Violation tracker	Two violation calls occur within five seconds.	Only the first is logged.
Violation tracker after cooldown	Violation occurs after cooldown.	A new CSV record can be created.

21. CURRENT IMPLEMENTATION STATUS
•	Dataset selection and preparation — completed.
•	YOLOv8 model training — completed.
•	Model validation — completed.
•	ONNX export — completed.
•	Local ONNX Runtime inference — completed.
•	Single-frame detector — completed.
•	Offline image testing — completed.
•	Offline video testing — completed.
•	Violation tracker — completed and tested.
•	Streamlit upload-video interface — implemented.
•	Streamlit webcam interface — implemented.
•	Tracker integration into the Streamlit app — next integration step.
•	requirements.txt and README.md — documentation files to finalize.

22. LIMITATIONS
•	The system detects the visual classes represented by the trained dataset; it does not understand the broader context of a person's situation.
•	Performance can vary with lighting, camera angle, distance, occlusion, image quality, and mask appearance.
•	The current preprocessing directly resizes frames to 640 × 640; this can distort aspect ratios compared with more advanced letterbox preprocessing.
•	The current violation metric accumulates detections across processed frames, so it is a detection count rather than a unique-person count.
•	The current tracker uses a simple time-based cooldown and does not track individual identities.
•	The system is a prototype and requires additional field testing before being used as an operational safety/compliance system.

23. FUTURE IMPROVEMENTS
•	Integrate ViolationTracker directly into app.py so video/webcam events are automatically logged.
•	Add unique-person tracking so the same person is not counted repeatedly across consecutive frames.
•	Improve preprocessing using aspect-ratio-preserving letterboxing.
•	Add configurable model paths and application settings.
•	Add CSV-log viewing and filtering inside the dashboard.
•	Add event screenshots or short clips for logged violations.
•	Support additional PPE classes such as helmets, safety vests, or other equipment if a suitable multi-class model is trained.
•	Add performance/FPS display and optional frame skipping for slower hardware.
•	Add a stronger reporting layer for date, time, source, class, and confidence.
•	Evaluate the model on footage representative of the intended deployment environment.

24. DEVELOPMENT ROADMAP
	Phase 1 — Dataset selection and preparation
	Phase 2 — YOLOv8 model training
	Phase 3 — Model validation
	Phase 4 — ONNX export for local deployment
	Phase 5 — Detector implementation
	Phase 6 — Image and video testing
	Phase 7 — Violation tracker implementation
	Phase 8 — Streamlit dashboard
	Phase 9 — Tracker/UI integration
	Phase 10 — Testing, optimization, documentation, and presentation

25. CODING STANDARDS AND DEVELOPMENT PRACTICES
•	Use object-oriented design where it makes responsibilities clearer.
•	Keep detector and tracker responsibilities independent.
•	Use descriptive variable and function names.
•	Keep functions small and testable.
•	Avoid unnecessary dependencies.
•	Use exception/error handling for camera and video operations.
•	Document important implementation decisions.
•	Test individual modules before integrating them into the UI.
•	Preserve the working ONNX-based deployment path when expanding the project.

26. SUCCESS CRITERIA
•	The trained model loads successfully.
•	Mask and No Mask instances can be detected in test images.
•	Bounding boxes and confidence labels are displayed correctly.
•	No Mask detections are counted as violations.
•	Uploaded videos can be processed frame-by-frame.
•	Webcam frames can be processed when camera access is available.
•	The violation tracker prevents repeated logging within its cooldown period.
•	The Streamlit interface provides a usable demonstration of the system.
•	The project remains modular and can be extended to additional PPE classes.

27. PROJECT SUMMARY
The completed Mask Detection System is a local computer-vision prototype built around a trained YOLOv8 model. The project separates model inference from violation logging and presents the result through a Streamlit interface. The training pipeline uses YOLOv8 and Google Colab, while the local deployment uses the exported ONNX model with ONNX Runtime. The system currently supports mask/no-mask detection, annotated image and video processing, webcam processing, confidence filtering, violation counting, and a cooldown-based CSV tracker. 
