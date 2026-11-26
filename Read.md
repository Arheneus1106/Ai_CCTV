# AI-Based Object Recognition in CCTV Video and Event Generation with Apache Kafka

This master’s thesis explores how a single CCTV camera at a construction-site gate can detect people and vehicles and report when they enter or leave the site. The goal is to turn a gate crossing into a structured event that another application can consume. For example, a car entering the site should produce one entry event, even if it appears in dozens of consecutive video frames.

## How it works

1. **Capture video.** OpenCV reads frames from a recorded video during development and from a live CCTV stream during deployment.
2. **Detect people and vehicles.** A lightweight YOLO model identifies objects in each frame and returns their locations and confidence scores.
3. **Track gate crossings.** ByteTrack links detections across frames and assigns each tracked object a temporary ID. The direction in which a track crosses a virtual line at the gate determines whether it is an entry or an exit.
4. **Publish events.** When a crossing is confirmed, a Python producer sends an event to Apache Kafka with the timestamp, camera ID, object class, track ID and crossing direction.

## Scope and evaluation

The first phase focuses on detecting people and vehicles in recorded footage. Later phases add live video, tracking and Kafka integration. A Raspberry Pi 5 is the target device for running the detection and tracking pipeline at the gate.

The evaluation compares detections and crossing events against manually labelled footage. It measures detection precision and recall, missed or duplicate crossing events, processing speed, memory usage and the delay between a gate crossing and the arrival of its Kafka event.

## Phase 1 — Recorded video detection

The baseline uses YOLO11n on the CPU to detect people, bicycles, cars, motorcycles, buses and trucks. [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)

Start with a short constant-frame-rate recording, such as a 15–30 second clip with both people and vehicles in view.

From the project directory, use Python 3.12 to set up the environment.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Save a recording as `videos/gate.mp4` or pass its existing path.

```sh
python detect_video.py videos/gate.mp4
```

The detector downloads the model weights if `yolo11n.pt` is missing. Results for this command are saved in `runs/gate/`.

- `annotated.mp4` shows the predicted boxes, class names and confidence scores.
- `detections.csv` has one row per detection with a zero-based frame index, estimated video time in seconds, class, confidence and bounding box in source-pixel coordinates. Video time is calculated from the nominal frame rate.

For the first thesis check, watch the annotated clip and manually label people and vehicles in a few frames. Compare those labels against the matching CSV rows and record missed objects and false detections. Use a new output directory to compare a different confidence threshold on the same clip.

```sh
python detect_video.py videos/gate.mp4 --confidence 0.40 --output-dir runs/gate-conf-040
```
