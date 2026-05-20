import time
import cv2
import sys

sys.path.append("../")
from utils import measure_distance, get_foot_position


class SpeedAndDistance_Estimator:
    def __init__(self, frame_rate=24, frame_window=5, min_delta_time=0.001, smoothing=0.6, movement_epsilon=0.02):
        self.frame_window = frame_window
        self.frame_rate = frame_rate
        self.min_delta_time = min_delta_time
        self.smoothing = smoothing
        self.movement_epsilon = movement_epsilon
        self.live_state = {}

    def _frame_timestamp(self, tracks, object_name, frame_num):
        timestamps = tracks.get("timestamps")
        if timestamps and frame_num < len(timestamps):
            return timestamps[frame_num]

        object_tracks = tracks.get(object_name, [])
        if frame_num < len(object_tracks):
            frame_tracks = object_tracks[frame_num]
            for track_info in frame_tracks.values():
                timestamp = track_info.get("timestamp")
                if timestamp is not None:
                    return timestamp

        return None

    def _safe_delta_time(self, start_frame, end_frame, start_time=None, end_time=None):
        frame_delta = end_frame - start_frame
        fallback_time = frame_delta / self.frame_rate if self.frame_rate > 0 and frame_delta > 0 else 0

        if start_time is not None and end_time is not None:
            timestamp_delta = end_time - start_time
            delta_time = timestamp_delta if timestamp_delta > self.min_delta_time else fallback_time
        else:
            delta_time = fallback_time

        return max(delta_time, self.min_delta_time)

    def _smooth_speed(self, previous_speed, current_speed):
        if previous_speed is None:
            return current_speed
        return (previous_speed * self.smoothing) + (current_speed * (1 - self.smoothing))

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        previous_speed = {}

        for object_name, object_tracks in tracks.items():
            if object_name in ("ball", "referees", "timestamps"):
                continue

            number_of_frames = len(object_tracks)
            if number_of_frames < 2:
                continue

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
                if last_frame <= frame_num:
                    continue

                start_time = self._frame_timestamp(tracks, object_name, frame_num)
                end_time = self._frame_timestamp(tracks, object_name, last_frame)
                time_elapsed = self._safe_delta_time(frame_num, last_frame, start_time, end_time)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id].get("position_transformed")
                    end_position = object_tracks[last_frame][track_id].get("position_transformed")

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    if distance_covered < self.movement_epsilon:
                        speed_km_per_hour = 0.0
                        distance_covered = 0.0
                    else:
                        speed_meters_per_second = distance_covered / time_elapsed
                        speed_km_per_hour = speed_meters_per_second * 3.6

                    total_distance.setdefault(object_name, {})
                    previous_speed.setdefault(object_name, {})
                    total_distance[object_name].setdefault(track_id, 0.0)

                    total_distance[object_name][track_id] += distance_covered
                    smoothed_speed = self._smooth_speed(previous_speed[object_name].get(track_id), speed_km_per_hour)
                    previous_speed[object_name][track_id] = smoothed_speed

                    for frame_num_batch in range(frame_num, last_frame + 1):
                        if track_id not in tracks[object_name][frame_num_batch]:
                            continue
                        tracks[object_name][frame_num_batch][track_id]["speed"] = smoothed_speed
                        tracks[object_name][frame_num_batch][track_id]["distance"] = total_distance[object_name][track_id]

    def update_live_tracks(self, tracks, frame_num=None, timestamp=None):
        if timestamp is None:
            timestamp = time.perf_counter()

        for object_name, object_tracks in tracks.items():
            if object_name in ("ball", "referees", "timestamps"):
                continue

            if isinstance(object_tracks, list):
                if not object_tracks:
                    continue
                current_tracks = object_tracks[-1]
                current_frame_num = len(object_tracks) - 1 if frame_num is None else frame_num
            else:
                current_tracks = object_tracks
                current_frame_num = 0 if frame_num is None else frame_num

            object_state = self.live_state.setdefault(object_name, {})
            active_track_ids = set(current_tracks.keys())

            for track_id, track_info in current_tracks.items():
                position = track_info.get("position_transformed")
                if position is None:
                    track_info["speed"] = object_state.get(track_id, {}).get("speed", 0.0)
                    track_info["distance"] = object_state.get(track_id, {}).get("distance", 0.0)
                    continue

                previous = object_state.get(track_id)
                if previous is None:
                    object_state[track_id] = {
                        "frame_num": current_frame_num,
                        "timestamp": timestamp,
                        "position": position,
                        "speed": 0.0,
                        "distance": 0.0,
                    }
                    track_info["speed"] = 0.0
                    track_info["distance"] = 0.0
                    continue

                repeated_frame = current_frame_num <= previous["frame_num"]
                time_elapsed = self._safe_delta_time(
                    previous["frame_num"],
                    current_frame_num,
                    previous.get("timestamp"),
                    timestamp,
                )
                if repeated_frame:
                    time_elapsed = max(timestamp - previous.get("timestamp", timestamp), self.min_delta_time)

                distance_delta = measure_distance(previous["position"], position)
                if distance_delta < self.movement_epsilon:
                    distance_delta = 0.0
                    speed_km_per_hour = 0.0
                else:
                    speed_km_per_hour = (distance_delta / time_elapsed) * 3.6

                smoothed_speed = self._smooth_speed(previous.get("speed"), speed_km_per_hour)
                total_distance = previous.get("distance", 0.0) + distance_delta

                object_state[track_id] = {
                    "frame_num": current_frame_num,
                    "timestamp": timestamp,
                    "position": position,
                    "speed": smoothed_speed,
                    "distance": total_distance,
                }
                track_info["speed"] = smoothed_speed
                track_info["distance"] = total_distance

            stale_track_ids = [track_id for track_id in object_state if track_id not in active_track_ids]
            for track_id in stale_track_ids:
                if timestamp - object_state[track_id].get("timestamp", timestamp) > 2.0:
                    del object_state[track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object_name, object_tracks in tracks.items():
                if object_name in ("ball", "referees", "timestamps"):
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" not in track_info:
                        continue

                    speed = track_info.get("speed")
                    distance = track_info.get("distance")
                    if speed is None or distance is None:
                        continue

                    bbox = track_info["bbox"]
                    position = list(get_foot_position(bbox))
                    position[1] += 40
                    position = tuple(map(int, position))

                    cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"{distance:.2f} m",
                        (position[0], position[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )
            output_frames.append(frame)

        return output_frames
