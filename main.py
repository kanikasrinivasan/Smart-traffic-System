#!/usr/bin/env python3
"""AI vehicle detection module for smart traffic management.

Reads a traffic video, runs YOLOv8 detection, maps detections to four directions
(NORTH/SOUTH/EAST/WEST) using frame quadrants, and emits machine-readable count
lines for downstream consumers (e.g., simulation.py).
"""

from __future__ import annotations

import argparse
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DIRECTIONS = ("NORTH", "SOUTH", "EAST", "WEST")
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class DetectionConfig:
    source: str = "night_traffic.mp4"
    weights: str = "yolov8n.pt"
    conf: float = 0.35
    interval_frames: int = 8
    show: bool = False
    mock: bool = False


def normalize_counts(counts: Dict[str, int]) -> Dict[str, int]:
    return {d: int(counts.get(d, 0)) for d in DIRECTIONS}


def format_counts(counts: Dict[str, int]) -> str:
    c = normalize_counts(counts)
    return f"COUNTS NORTH={c['NORTH']} SOUTH={c['SOUTH']} EAST={c['EAST']} WEST={c['WEST']}"


def emit_counts(counts: Dict[str, int]) -> bool:
    try:
        print(format_counts(counts), flush=True)
        return True
    except BrokenPipeError:
        return False


def direction_from_point(x: float, y: float, width: float, height: float) -> str:
    """Map a box center to a cardinal direction using frame quadrants.

    Top half maps to NORTH/SOUTH split by left/right,
    Bottom half maps to WEST/EAST split by left/right.
    """
    top = y < (height / 2)
    left = x < (width / 2)

    if top and left:
        return "NORTH"
    if top and not left:
        return "SOUTH"
    if not top and not left:
        return "EAST"
    return "WEST"


def run_mock_stream() -> Iterable[Dict[str, int]]:
    """Generate plausible traffic counts without CV dependencies."""
    base = {"NORTH": 2, "SOUTH": 2, "EAST": 1, "WEST": 1}
    while True:
        swing = random.choice(DIRECTIONS)
        counts = {k: max(0, v + random.randint(-1, 2)) for k, v in base.items()}
        counts[swing] += random.randint(1, 4)
        base = counts
        yield counts


def run_yolo(config: DetectionConfig) -> None:
    try:
        import cv2  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing required packages. Install with: pip install ultralytics opencv-python"
        ) from exc

    source_path = Path(config.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Video source not found: {source_path}")

    model = YOLO(config.weights)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if frame_idx % max(config.interval_frames, 1) != 0:
                if config.show:
                    cv2.imshow("Smart Traffic Detection", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                continue

            h, w = frame.shape[:2]
            counts = {d: 0 for d in DIRECTIONS}

            results = model.predict(frame, conf=config.conf, verbose=False)
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue

                for box in boxes:
                    cls_idx = int(box.cls[0].item())
                    cls_name = model.names.get(cls_idx, str(cls_idx)).lower()
                    if cls_name not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    cx, cy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    direction = direction_from_point(cx, cy, w, h)
                    counts[direction] += 1

                    if config.show:
                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"{cls_name}:{direction}",
                            (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

            if not emit_counts(counts):
                break

            if config.show:
                cv2.imshow("Smart Traffic Detection", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        cap.release()
        if config.show:
            cv2.destroyAllWindows()


def parse_args(argv: List[str]) -> DetectionConfig:
    p = argparse.ArgumentParser(description="Run YOLOv8-based vehicle counting by direction.")
    p.add_argument("--source", default="night_traffic.mp4", help="Path to video source")
    p.add_argument("--weights", default="yolov8n.pt", help="Path to YOLO weights")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--interval-frames", type=int, default=8, help="Process every Nth frame")
    p.add_argument("--show", action="store_true", help="Show OpenCV preview window")
    p.add_argument("--mock", action="store_true", help="Use synthetic counts (no YOLO/OpenCV)")
    args = p.parse_args(argv)
    return DetectionConfig(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        interval_frames=args.interval_frames,
        show=args.show,
        mock=args.mock,
    )


def main(argv: List[str] | None = None) -> int:
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    config = parse_args(argv or sys.argv[1:])

    if config.mock:
        for counts in run_mock_stream():
            if not emit_counts(counts):
                break
            time.sleep(0.6)
        return 0

    run_yolo(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
