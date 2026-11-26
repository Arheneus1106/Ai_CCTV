
import argparse
import csv
import math
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter


# COCO IDs for person, bicycle, car, motorcycle, bus, and truck.
CLASS_IDS = [0, 1, 2, 3, 5, 7]


def detect_video(source: Path, output: Path, confidence: float) -> None:
    import cv2
    from ultralytics import YOLO

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        with ExitStack() as resources:
            capture = cv2.VideoCapture(str(source))
            resources.callback(capture.release)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {source}")

            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise RuntimeError(f"Could not determine the frame rate: {source}")

            has_frame, frame = capture.read()
            if not has_frame:
                raise RuntimeError(f"Could not read the first frame: {source}")

            height, width = frame.shape[:2]
            pad_bottom, pad_right = height % 2, width % 2
            writer = cv2.VideoWriter(
                str(staging / "annotated.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width + pad_right, height + pad_bottom),
            )
            resources.callback(writer.release)
            if not writer.isOpened():
                raise RuntimeError("Could not open the annotated MP4 for writing")

            csv_file = resources.enter_context(
                (staging / "detections.csv").open("w", encoding="utf-8", newline="")
            )
            detections = csv.writer(csv_file)
            detections.writerow(
                ["frame_index", "time_s", "class", "confidence", "x1", "y1", "x2", "y2"]
            )

            model = YOLO("yolo11n.pt")
            frame_index = 0
            detection_count = 0
            started_at = perf_counter()
            while has_frame:
                result = model.predict(
                    frame,
                    conf=confidence,
                    classes=CLASS_IDS,
                    imgsz=640,
                    device="cpu",
                    verbose=False,
                )[0]
                for x1, y1, x2, y2, score, class_id in result.boxes.data.tolist():
                    detections.writerow(
                        [
                            frame_index,
                            f"{frame_index / fps:.6f}",
                            result.names[int(class_id)],
                            f"{score:.6f}",
                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2),
                        ]
                    )
                    detection_count += 1

                annotated = result.plot()
                # Pad odd dimensions so MP4 encoding keeps the full image.
                if pad_bottom or pad_right:
                    annotated = cv2.copyMakeBorder(
                        annotated, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT
                    )
                writer.write(annotated)
                frame_index += 1
                has_frame, frame = capture.read()

        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output path already exists: {output}")
        staging.rename(output)
    elapsed = perf_counter() - started_at
    print(
        f"Processed {frame_index} frames with {detection_count} detections "
        f"({frame_index / elapsed:.1f} frames/s)."
    )
    print(f"Saved annotated.mp4 and detections.csv in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="recorded video file")
    parser.add_argument("--output-dir", type=Path, help="new directory for the results")
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="minimum confidence (default: 0.25)"
    )
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video file does not exist: {args.video}")
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")

    output = args.output_dir or Path("runs") / args.video.stem
    if output.exists() or output.is_symlink():
        parser.error(f"Output path already exists: {output}. Choose a new --output-dir.")

    try:
        detect_video(args.video, output, args.confidence)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
