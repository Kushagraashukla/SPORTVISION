import argparse
import json
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

try:
    from analytics import PlayerAnalytics
    from heatmap_generator import generate_heatmap
    ANALYTICS_IMPORT_ERROR = None
except Exception as exc:
    PlayerAnalytics = None
    generate_heatmap = None
    ANALYTICS_IMPORT_ERROR = exc

from player_ball_assigner import PlayerBallAssigner
from team_assigner import TeamAssigner
from trackers import Tracker
from utils import get_foot_position, measure_distance
from view_transformer import ViewTransformer


DASHBOARD_STATS_URL = "http://127.0.0.1:8000/stats"
ANALYTICS_SEND_INTERVAL_FRAMES = 30
ANALYTICS_TIMEOUT_SECONDS = 0.2
ANALYTICS_DISABLE_FRAMES = 30


def setup_analytics_logging():
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("passive_analytics")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(Path("logs") / "analytics.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    return logger


def bbox_center(bbox):
    try:
        return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)
    except Exception:
        return None


def safe_dashboard_post(payload, logger):
    try:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            DASHBOARD_STATS_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=ANALYTICS_TIMEOUT_SECONDS):
            return
    except Exception as exc:
        logger.warning("Dashboard stats post skipped: %s", exc)


def send_dashboard_stats_async(payload, logger):
    try:
        thread = threading.Thread(target=safe_dashboard_post, args=(payload, logger), daemon=True)
        thread.start()
    except Exception as exc:
        logger.warning("Dashboard stats thread skipped: %s", exc)


def build_dashboard_payload(frame_num, fps, analytics, possession_seconds):
    players = []
    if analytics is None:
        return {
            "match_id": "default_match",
            "camera_id": "video_pipeline",
            "frame_number": frame_num,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "players": players,
            "match": {"fps": fps},
        }

    for player_id, stats in analytics.get_all_stats().items():
        touches = stats.get("touch_count", 0)
        possession_time = possession_seconds.get(player_id, 0.0)
        activity_score = min(
            100.0,
            (float(stats.get("distance_km", 0.0)) * 28.0) + (float(touches) * 4.0) + (float(possession_time) * 0.4),
        )
        players.append(
            {
                "player_id": str(stats.get("player_id", player_id)),
                "team": None if stats.get("team") is None else str(stats.get("team")),
                "distance_km": float(stats.get("distance_km", 0.0)),
                "current_speed_kmh": float(stats.get("current_speed_kmh", 0.0)),
                "avg_speed_kmh": float(stats.get("avg_speed_kmh", 0.0)),
                "ball_touches": int(touches),
                "possession_time_sec": round(float(possession_time), 2),
                "activity_score": round(activity_score, 2),
                "confidence_score": float(stats.get("confidence_score", 0.0)),
            }
        )

    return {
        "match_id": "default_match",
        "camera_id": "video_pipeline",
        "frame_number": frame_num,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "players": players,
        "match": {"fps": fps},
    }


def generate_player_heatmaps(analytics, field_width, field_height, logger):
    heatmaps = {}
    if analytics is None or generate_heatmap is None:
        logger.warning("Heatmap generation skipped because passive analytics is unavailable.")
        return heatmaps

    for player_id, positions in analytics.get_heatmap_history().items():
        try:
            path = generate_heatmap(player_id, positions, field_width, field_height)
            heatmaps[str(player_id)] = path
        except Exception as exc:
            logger.error("Heatmap generation skipped for player %s: %s", player_id, exc)
            heatmaps[str(player_id)] = None
    return heatmaps


def write_analytics_report(analytics, heatmaps, output_path, logger):
    try:
        players = []
        if analytics is not None:
            for stats in analytics.get_all_stats().values():
                safe_stats = dict(stats)
                safe_stats.pop("sprint_count", None)
                players.append(safe_stats)
        report = {
            "output_video": output_path,
            "players": players,
            "heatmaps": heatmaps,
        }
        with open("analytics_report.json", "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)
    except Exception as exc:
        logger.error("Analytics report write skipped: %s", exc)


def default_output_path(input_path):
    input_name = Path(input_path).stem if input_path else "football_analysis"
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in input_name)
    return str(Path("output_videos") / f"{safe_name}_output.avi")


def open_capture(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    return cap


def make_writer(output_path, fps, width, height):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps if fps and fps > 0 else 24, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    return writer


def resize_for_preview(frame, preview_width):
    if not preview_width or frame.shape[1] <= preview_width:
        return frame
    scale = preview_width / frame.shape[1]
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (preview_width, height), interpolation=cv2.INTER_AREA)


def resolve_device(requested_device):
    if requested_device != "auto":
        return requested_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def detections_for_frame(tracker, frame, confidence, device):
    detection = tracker.model.predict(frame, conf=confidence, device=device, verbose=False)[0]
    cls_names = detection.names
    cls_names_inv = {value: key for key, value in cls_names.items()}
    detections = sv.Detections.from_ultralytics(detection)

    if "goalkeeper" in cls_names_inv and "player" in cls_names_inv:
        for object_ind, class_id in enumerate(detections.class_id):
            if cls_names[class_id] == "goalkeeper":
                detections.class_id[object_ind] = cls_names_inv["player"]

    return detections, cls_names, cls_names_inv


def build_tracks_for_frame(tracker, detections, cls_names, cls_names_inv):
    tracks = {"players": {}, "referees": {}, "ball": {}}
    detections_with_tracks = tracker.tracker.update_with_detections(detections)

    for frame_detection in detections_with_tracks:
        bbox = frame_detection[0].tolist()
        cls_id = frame_detection[3]
        track_id = frame_detection[4]

        if cls_id == cls_names_inv.get("player"):
            tracks["players"][track_id] = {"bbox": bbox}
        elif cls_id == cls_names_inv.get("referee"):
            tracks["referees"][track_id] = {"bbox": bbox}

    for frame_detection in detections:
        bbox = frame_detection[0].tolist()
        cls_id = frame_detection[3]
        if cls_id == cls_names_inv.get("ball"):
            tracks["ball"][1] = {"bbox": bbox}

    return tracks


def update_speed_and_distance(tracks, speed_state, view_transformer, frame_num, fps):
    for track_id, player in tracks["players"].items():
        foot_position = np.array(get_foot_position(player["bbox"]))
        transformed = view_transformer.transform_point(foot_position)
        if transformed is None:
            continue

        position_m = transformed.squeeze().tolist()
        previous = speed_state.get(track_id)
        if previous is None:
            speed_state[track_id] = {
                "frame": frame_num,
                "position": position_m,
                "speed": 0.0,
                "distance": 0.0,
            }
            player["speed"] = 0.0
            player["distance"] = 0.0
            continue

        frame_delta = max(frame_num - previous["frame"], 1)
        time_delta = frame_delta / fps if fps and fps > 0 else frame_delta / 24
        distance_delta = measure_distance(previous["position"], position_m)
        total_distance = previous["distance"] + distance_delta
        speed_kmh = (distance_delta / time_delta) * 3.6 if time_delta > 0 else previous["speed"]

        smooth_speed = previous["speed"] * 0.6 + speed_kmh * 0.4
        speed_state[track_id] = {
            "frame": frame_num,
            "position": position_m,
            "speed": smooth_speed,
            "distance": total_distance,
        }
        player["speed"] = smooth_speed
        player["distance"] = total_distance


def try_assign_teams(team_assigner, frame, player_tracks, teams_ready):
    if not teams_ready and len(player_tracks) >= 2:
        try:
            team_assigner.assign_team_color(frame, player_tracks)
            teams_ready = True
        except Exception as exc:
            print(f"Team color setup skipped on this frame: {exc}")

    if teams_ready:
        for player_id, track in player_tracks.items():
            try:
                team = team_assigner.get_player_team(frame, track["bbox"], player_id)
                track["team"] = team
                track["team_color"] = team_assigner.team_colors[team]
            except Exception:
                track["team"] = 1
                track["team_color"] = (0, 0, 255)
    else:
        for track in player_tracks.values():
            track["team"] = 1
            track["team_color"] = (0, 0, 255)

    return teams_ready


def draw_ball_control(frame, team_ball_control):
    height, width = frame.shape[:2]
    box_width = min(520, width - 40)
    x1 = max(20, width - box_width - 20)
    y1 = max(20, height - 125)
    x2 = width - 20
    y2 = height - 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    controls = np.array(team_ball_control)
    team_1_frames = controls[controls == 1].shape[0]
    team_2_frames = controls[controls == 2].shape[0]
    total = max(team_1_frames + team_2_frames, 1)
    team_1 = team_1_frames / total
    team_2 = team_2_frames / total

    font_scale = 0.7 if width < 1280 else 0.9
    cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.1f}%", (x1 + 15, y1 + 42),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.1f}%", (x1 + 15, y1 + 82),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    return frame


def draw_frame(tracker, frame, tracks, team_ball_control):
    for track_id, player in tracks["players"].items():
        color = player.get("team_color", (0, 0, 255))
        frame = tracker.draw_ellipse(frame, player["bbox"], color, track_id)
        if player.get("has_ball", False):
            frame = tracker.draw_traingle(frame, player["bbox"], (0, 0, 255))
        if "speed" in player and "distance" in player:
            position = list(get_foot_position(player["bbox"]))
            position[1] += 38
            cv2.putText(frame, f"{player['speed']:.1f} km/h", tuple(position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(frame, f"{player['distance']:.1f} m", (position[0], position[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    for referee in tracks["referees"].values():
        frame = tracker.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

    for ball in tracks["ball"].values():
        frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

    return draw_ball_control(frame, team_ball_control)


def main():
    parser = argparse.ArgumentParser(
        description="Run football analysis directly from any complete video path with live preview and output saving."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Complete video path. Example: C:\\Users\\OMEN\\Downloads\\match.mp4",
    )
    parser.add_argument(
        "--input",
        dest="input_option",
        default=None,
        help="Complete video path. Kept for compatibility with older commands.",
    )
    parser.add_argument("--output", default=None, help="Output video path. Default: output_videos/<input_name>_output.avi")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path. Default: models/best.pt")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold. Default: 0.1")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, etc. Default: auto",
    )
    parser.add_argument("--no-display", action="store_true", help="Save output without opening the live preview window.")
    parser.add_argument("--preview-width", type=int, default=1280, help="Preview window width. Default: 1280")
    args = parser.parse_args()

    input_path = args.input_option or args.input_path
    if not input_path:
        default_input = Path("input_videos") / "08fd33_4.mp4"
        if default_input.exists():
            input_path = str(default_input)
        else:
            input_path = input("Paste complete video path: ").strip().strip('"')

    input_path = input_path.strip('"')
    output_path = args.output or default_output_path(input_path)

    cap = open_capture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = Tracker(args.model)
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()
    view_transformer = ViewTransformer()
    writer = make_writer(output_path, fps, width, height)
    device = resolve_device(args.device)

    team_ball_control = []
    speed_state = {}
    teams_ready = False
    frame_num = 0
    try:
        analytics_logger = setup_analytics_logging()
    except Exception:
        analytics_logger = logging.getLogger("passive_analytics_fallback")
        analytics_logger.addHandler(logging.NullHandler())
    try:
        analytics = PlayerAnalytics(sprint_speed_threshold_kmh=9999.0) if PlayerAnalytics is not None else None
    except Exception as exc:
        analytics = None
        analytics_logger.error("Passive analytics initialization skipped: %s", exc)
    analytics_previous_positions = {}
    analytics_possession_seconds = {}
    analytics_disabled_until_frame = -1
    if ANALYTICS_IMPORT_ERROR is not None:
        analytics_logger.error("Passive analytics unavailable at import time: %s", ANALYTICS_IMPORT_ERROR)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    if device == "cpu":
        print("GPU is not active. Install CUDA-enabled PyTorch to use NVIDIA GPU.")
    print("Live preview is running. Press q in the preview window to stop early.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections, cls_names, cls_names_inv = detections_for_frame(tracker, frame, args.conf, device)
            tracks = build_tracks_for_frame(tracker, detections, cls_names, cls_names_inv)

            teams_ready = try_assign_teams(team_assigner, frame, tracks["players"], teams_ready)
            update_speed_and_distance(tracks, speed_state, view_transformer, frame_num, fps)

            assigned_player = -1
            if tracks["ball"]:
                ball_bbox = tracks["ball"][1]["bbox"]
                assigned_player = player_assigner.assign_ball_to_player(tracks["players"], ball_bbox)

            if assigned_player != -1 and assigned_player in tracks["players"]:
                tracks["players"][assigned_player]["has_ball"] = True
                team_ball_control.append(tracks["players"][assigned_player].get("team", 1))
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

            if analytics is not None and frame_num >= analytics_disabled_until_frame:
                try:
                    ball_position = None
                    if tracks["ball"]:
                        ball_position = bbox_center(tracks["ball"][1]["bbox"])

                    for player_id, player in tracks["players"].items():
                        current_position = bbox_center(player.get("bbox"))
                        previous_position = analytics_previous_positions.get(player_id)
                        analytics.update_player(
                            player_id=player_id,
                            current_position=current_position,
                            previous_position=previous_position,
                            frame_number=frame_num,
                            fps=fps,
                            ball_position=ball_position,
                            team=player.get("team"),
                        )
                        if current_position is not None:
                            analytics_previous_positions[player_id] = current_position

                    if assigned_player != -1:
                        analytics_possession_seconds[assigned_player] = analytics_possession_seconds.get(
                            assigned_player,
                            0.0,
                        ) + (1.0 / fps if fps and fps > 0 else 1.0 / 24.0)

                    if frame_num % ANALYTICS_SEND_INTERVAL_FRAMES == 0:
                        payload = build_dashboard_payload(frame_num, fps, analytics, analytics_possession_seconds)
                        send_dashboard_stats_async(payload, analytics_logger)
                except Exception as exc:
                    analytics_disabled_until_frame = frame_num + ANALYTICS_DISABLE_FRAMES
                    analytics_logger.error(
                        "Passive analytics disabled until frame %s after error: %s",
                        analytics_disabled_until_frame,
                        exc,
                    )

            annotated_frame = draw_frame(tracker, frame.copy(), tracks, team_ball_control)
            writer.write(annotated_frame)

            frame_num += 1
            if frame_num == 1 or frame_num % 25 == 0:
                suffix = f"/{total_frames}" if total_frames > 0 else ""
                possession = f", player {assigned_player} has ball" if assigned_player != -1 else ""
                print(f"Processed frame {frame_num}{suffix}{possession}")

            if not args.no_display:
                preview_frame = resize_for_preview(annotated_frame, args.preview_width)
                cv2.imshow("Football Analysis - Live Output", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped early by user.")
                    break
    finally:
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    try:
        heatmaps = generate_player_heatmaps(analytics, width, height, analytics_logger)
        write_analytics_report(analytics, heatmaps, output_path, analytics_logger)
        send_dashboard_stats_async(build_dashboard_payload(frame_num, fps, analytics, analytics_possession_seconds), analytics_logger)
    except Exception as exc:
        analytics_logger.error("Final passive analytics export skipped: %s", exc)

    print(f"Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
