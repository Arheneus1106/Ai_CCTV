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