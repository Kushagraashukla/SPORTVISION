import argparse
import os
import time

import cv2
import numpy as np
import supervision as sv

from speed_and_distance_estimator import SpeedAndDistance_Estimator
from trackers import Tracker
from utils import get_foot_position
from view_transformer import ViewTransformer


def parse_source(source):
    try:
        return int(source)
    except ValueError:
        return source


def open_capture(source):
    cap = cv2.VideoCapture(parse_source(source))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def is_file_source(source):
    return not source.isdigit() and os.path.exists(source)


def resolve_device(requested_device):
    if requested_device != "auto":
        return requested_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def add_live_positions(tracks, view_transformer):
    for player in tracks["players"].values():
        position = np.array(get_foot_position(player["bbox"]))
        transformed = view_transformer.transform_point(position)
        player["position_transformed"] = transformed.squeeze().tolist() if transformed is not None else None


def draw_speed_and_distance(frame, tracks):
    for player in tracks["players"].values():
        speed = player.get("speed")
        distance = player.get("distance")
        if speed is None or distance is None:
            continue

        position = list(get_foot_position(player["bbox"]))
        position[1] += 38
        cv2.putText(frame, f"{speed:.1f} km/h", tuple(position), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(
            frame,
            f"{distance:.1f} m",
            (position[0], position[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )
    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Run live football object detection and tracking.")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, RTSP URL, or stream URL. Default: 0",
    )
    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="YOLO model path. Default: models/best.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Detection confidence threshold. Default: 0.1",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output video path, for example output_videos/live_output.avi",
    )
    parser.add_argument(
        "--max-read-failures",
        type=int,
        default=0,
        help="Consecutive failed reads before closing. Use 0 to keep retrying forever. Default: 0",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=0.5,
        help="Seconds to wait before retrying/reconnecting after a failed live read. Default: 0.5",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without opening the preview window.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, etc. Default: auto",
    )
    args = parser.parse_args()

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    tracker = Tracker(args.model)
    view_transformer = ViewTransformer()
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    device = resolve_device(args.device)
    writer = None
    read_failures = 0
    source_is_file = is_file_source(args.source)
    frame_num = 0
    last_frame_time = None
    smoothed_fps = 0.0

    print(f"Device: {device}")
    if device == "cpu":
        print("GPU is not active. Install CUDA-enabled PyTorch to use NVIDIA GPU.")

    while True:
        ret, frame = cap.read()
        frame_time = time.perf_counter()
        if not ret:
            if source_is_file:
                print("Finished reading recorded video.")
                break

            read_failures += 1
            if args.max_read_failures and read_failures >= args.max_read_failures:
                print(f"Stopping after {read_failures} failed reads from live source.")
                break

            print(f"Live frame not received, reconnecting... ({read_failures})")
            cap.release()
            time.sleep(args.reconnect_delay)
            cap = open_capture(args.source)

            if not args.no_display and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        read_failures = 0
        if last_frame_time is not None:
            delta_time = max(frame_time - last_frame_time, 0.001)
            current_fps = 1.0 / delta_time
            smoothed_fps = current_fps if smoothed_fps == 0 else (smoothed_fps * 0.9) + (current_fps * 0.1)
        last_frame_time = frame_time

        detection = tracker.model.predict(frame, conf=args.conf, device=device, verbose=False)[0]
        cls_names = detection.names
        cls_names_inv = {value: key for key, value in cls_names.items()}
        detections = sv.Detections.from_ultralytics(detection)

        if "goalkeeper" in cls_names_inv and "player" in cls_names_inv:
            for object_ind, class_id in enumerate(detections.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detections.class_id[object_ind] = cls_names_inv["player"]

        detections_with_tracks = tracker.tracker.update_with_detections(detections)
        current_tracks = {
            "players": {},
            "referees": {},
            "ball": {},
        }

        for frame_detection in detections_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]

            if cls_id == cls_names_inv.get("player"):
                current_tracks["players"][track_id] = {"bbox": bbox, "timestamp": frame_time}
            elif cls_id == cls_names_inv.get("referee"):
                current_tracks["referees"][track_id] = {"bbox": bbox, "timestamp": frame_time}

        for frame_detection in detections:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            if cls_id == cls_names_inv.get("ball"):
                current_tracks["ball"][1] = {"bbox": bbox, "timestamp": frame_time}

        add_live_positions(current_tracks, view_transformer)
        speed_and_distance_estimator.update_live_tracks(current_tracks, frame_num=frame_num, timestamp=frame_time)

        for track_id, player in current_tracks["players"].items():
            frame = tracker.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)

        for referee in current_tracks["referees"].values():
            frame = tracker.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

        for ball in current_tracks["ball"].values():
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        frame = draw_speed_and_distance(frame, current_tracks)
        frame = draw_fps(frame, smoothed_fps)

        if args.output and writer is None:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = cap.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(args.output, fourcc, fps if fps and fps > 0 else 24, (width, height))

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Football Analysis Live", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_num += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
