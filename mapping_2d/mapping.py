import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0
DEFAULT_FRAME_WIDTH = 1920.0
DEFAULT_FRAME_HEIGHT = 1080.0


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def as_point_array(points: List[List[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Point arrays must have shape Nx2.")
    return array


def normalize_homography(matrix: np.ndarray) -> np.ndarray:
    if matrix[2, 2] == 0:
        return matrix
    return matrix / matrix[2, 2]


def get_frame_calibration(
    calibration_data: Dict[str, Any],
    frame_name: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    frames = calibration_data.get("frames", {})
    if frame_name in frames:
        return frames[frame_name], "frame"
    if "default" in calibration_data:
        return calibration_data["default"], "default"
    if frame_name in calibration_data:
        return calibration_data[frame_name], "frame"
    return None, "missing"


def estimate_homography(
    image_points: List[List[float]],
    pitch_points: List[List[float]],
    ransac_threshold: float,
) -> Tuple[np.ndarray, List[int], float]:
    src = as_point_array(image_points)
    dst = as_point_array(pitch_points)

    if len(src) < 4 or len(dst) < 4 or len(src) != len(dst):
        raise ValueError("Each calibration entry needs at least 4 matching point pairs.")

    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
    if homography is None or mask is None:
        raise ValueError("OpenCV could not estimate a homography from the provided points.")

    homography = normalize_homography(homography)
    inlier_mask = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    inlier_errors = errors[inlier_mask]
    reprojection_error = float(np.mean(inlier_errors)) if len(inlier_errors) else float(np.mean(errors))
    inlier_indices = [int(index) for index, is_inlier in enumerate(inlier_mask) if is_inlier]
    return homography, inlier_indices, reprojection_error


def smooth_homography(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    alpha: float,
) -> np.ndarray:
    current = normalize_homography(current)
    if previous is None:
        return current
    previous = normalize_homography(previous)
    blended = alpha * previous + (1.0 - alpha) * current
    return normalize_homography(blended)


def project_point(point: List[float], homography: np.ndarray) -> List[float]:
    src = np.asarray([[point]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)[0, 0]
    return [round(float(dst[0]), 3), round(float(dst[1]), 3)]


def point_inside_pitch(point: List[float], pitch_length: float, pitch_width: float) -> bool:
    x, y = point
    return 0.0 <= x <= pitch_length and 0.0 <= y <= pitch_width


def infer_frame_size(
    detections_by_frame: Dict[str, Any],
    fallback_width: float = DEFAULT_FRAME_WIDTH,
    fallback_height: float = DEFAULT_FRAME_HEIGHT,
) -> Tuple[float, float]:
    max_x = 0.0
    max_y = 0.0

    for frame_detections in detections_by_frame.values():
        for detection in frame_detections:
            bbox = detection.get("bbox_image")
            if bbox and len(bbox) == 4:
                max_x = max(max_x, float(bbox[2]))
                max_y = max(max_y, float(bbox[3]))
            foot_point = detection.get("foot_point")
            if foot_point and len(foot_point) == 2:
                max_x = max(max_x, float(foot_point[0]))
                max_y = max(max_y, float(foot_point[1]))

    width = max(max_x, fallback_width)
    height = max(max_y, fallback_height)
    return width, height


def infer_images_dir(detections_path: Path) -> Optional[Path]:
    candidates = [
        detections_path.parent.parent.parent / "img1",
        detections_path.parent.parent / "img1",
        detections_path.parent / "img1",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def estimate_auto_bounds(
    frame_detections: List[Dict[str, Any]],
    frame_width: float,
    frame_height: float,
) -> np.ndarray:
    if not frame_detections:
        return np.asarray([0.0, frame_width, 0.0, frame_height], dtype=np.float32)

    foot_points = np.asarray([d["foot_point"] for d in frame_detections], dtype=np.float32)
    xs = foot_points[:, 0]
    ys = foot_points[:, 1]

    left = float(np.percentile(xs, 5))
    right = float(np.percentile(xs, 95))
    top = float(np.percentile(ys, 5))
    bottom = float(np.percentile(ys, 95))

    width_pad = max(40.0, 0.15 * max(right - left, 1.0))
    height_pad = max(30.0, 0.12 * max(bottom - top, 1.0))

    left = max(0.0, left - width_pad)
    right = min(frame_width, right + width_pad)
    top = max(0.0, top - height_pad)
    bottom = min(frame_height, bottom + height_pad)

    if right - left < 1.0:
        left, right = 0.0, frame_width
    if bottom - top < 1.0:
        top, bottom = 0.0, frame_height

    return np.asarray([left, right, top, bottom], dtype=np.float32)


def smooth_bounds(previous: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
    if previous is None:
        return current
    return alpha * previous + (1.0 - alpha) * current


def detect_view_cues(image_path: Optional[Path]) -> Dict[str, float]:
    if image_path is None or not image_path.exists():
        return {
            "left_box_score": 0.0,
            "right_box_score": 0.0,
            "center_line_score": 0.0,
            "center_circle_score": 0.0,
        }

    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "left_box_score": 0.0,
            "right_box_score": 0.0,
            "center_line_score": 0.0,
            "center_circle_score": 0.0,
        }

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    edges = cv2.Canny(white_mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=25,
        minLineLength=max(int(width * 0.06), 30),
        maxLineGap=20,
    )

    left_vertical = 0.0
    right_vertical = 0.0
    left_horizontal = 0.0
    right_horizontal = 0.0
    center_vertical = 0.0

    if lines is not None:
        for segment in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in segment]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < 1.0:
                continue

            angle = abs(np.degrees(np.arctan2(dy, dx)))
            x_mid = (x1 + x2) / 2.0
            y_mid = (y1 + y2) / 2.0

            if 75.0 <= angle <= 105.0:
                if 0.04 * width <= x_mid <= 0.24 * width and 0.18 * height <= y_mid <= 0.92 * height:
                    left_vertical += length
                elif 0.76 * width <= x_mid <= 0.96 * width and 0.18 * height <= y_mid <= 0.92 * height:
                    right_vertical += length
                elif 0.42 * width <= x_mid <= 0.58 * width and 0.08 * height <= y_mid <= 0.95 * height:
                    center_vertical += length
            elif angle <= 18.0 or angle >= 162.0:
                if 0.02 * width <= x_mid <= 0.40 * width and 0.18 * height <= y_mid <= 0.88 * height:
                    left_horizontal += length
                elif 0.60 * width <= x_mid <= 0.98 * width and 0.18 * height <= y_mid <= 0.88 * height:
                    right_horizontal += length

    center_circle_score = 0.0
    blurred = cv2.GaussianBlur(white_mask, (9, 9), 2.0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(int(min(width, height) * 0.12), 30),
        param1=100,
        param2=14,
        minRadius=max(int(min(width, height) * 0.10), 20),
        maxRadius=max(int(min(width, height) * 0.30), 40),
    )
    if circles is not None:
        for circle in circles[0]:
            x, y, radius = [float(v) for v in circle]
            if 0.25 * width <= x <= 0.75 * width and 0.20 * height <= y <= 0.85 * height:
                center_circle_score = max(center_circle_score, radius * 2.0)

    return {
        "left_box_score": round(left_vertical + 0.7 * left_horizontal, 3),
        "right_box_score": round(right_vertical + 0.7 * right_horizontal, 3),
        "center_line_score": round(center_vertical, 3),
        "center_circle_score": round(center_circle_score, 3),
    }


def estimate_auto_pitch_window(
    frame_detections: List[Dict[str, Any]],
    image_bounds: np.ndarray,
    frame_width: float,
    pitch_length: float,
    previous_window: Optional[np.ndarray],
    view_cues: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    left, right, _, _ = [float(v) for v in image_bounds]
    visible_width_ratio = float(np.clip((right - left) / max(frame_width, 1.0), 0.15, 1.0))
    window_length = float(np.clip(25.0 + visible_width_ratio * 65.0, 32.0, 85.0))

    if frame_detections:
        foot_points = np.asarray([d["foot_point"] for d in frame_detections], dtype=np.float32)
        image_center_ratio = float(np.median(foot_points[:, 0]) / max(frame_width, 1.0))
    else:
        image_center_ratio = 0.5

    left_box_score = float((view_cues or {}).get("left_box_score", 0.0))
    right_box_score = float((view_cues or {}).get("right_box_score", 0.0))
    center_line_score = float((view_cues or {}).get("center_line_score", 0.0))
    center_circle_score = float((view_cues or {}).get("center_circle_score", 0.0))
    strong_right_box = right_box_score > max(260.0, left_box_score * 1.15)
    strong_left_box = left_box_score > max(260.0, right_box_score * 1.15)
    center_visible = center_circle_score > 80.0 or center_line_score > 180.0

    strong_frame_cues = False
    if strong_left_box:
        anchor_ratio = 0.26 if center_visible else 0.18
        window_length = min(window_length, 62.0 if center_visible else 48.0)
        anchor_weight = 1.0
        strong_frame_cues = True
    elif strong_right_box:
        anchor_ratio = 0.74 if center_visible else 0.82
        window_length = min(window_length, 62.0 if center_visible else 48.0)
        anchor_weight = 1.0
        strong_frame_cues = True
    elif center_visible:
        anchor_ratio = 0.50
        window_length = min(window_length, 72.0)
        anchor_weight = 0.95
        strong_frame_cues = True
    elif image_center_ratio < 0.38:
        anchor_ratio = 0.32
        anchor_weight = 0.45
    elif image_center_ratio > 0.62:
        anchor_ratio = 0.68
        anchor_weight = 0.45
    else:
        anchor_ratio = 0.50
        anchor_weight = 0.45

    anchor_center = anchor_ratio * pitch_length

    if previous_window is None:
        pitch_center = anchor_center
    else:
        prev_center = float(previous_window[0])
        prev_length = float(previous_window[1])
        image_offset = image_center_ratio - 0.5
        predicted_center = prev_center + image_offset * prev_length * 0.75
        if strong_frame_cues:
            pitch_center = anchor_center
        else:
            pitch_center = (1.0 - anchor_weight) * predicted_center + anchor_weight * anchor_center

    min_center = window_length / 2.0
    max_center = pitch_length - window_length / 2.0
    pitch_center = float(np.clip(pitch_center, min_center, max_center))
    return np.asarray(
        [pitch_center, window_length, anchor_ratio, image_center_ratio, 1.0 if strong_frame_cues else 0.0],
        dtype=np.float32,
    )


def smooth_pitch_window(previous: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
    if previous is None:
        return current
    if len(current) >= 5 and float(current[4]) > 0.5:
        return current
    smoothed = alpha * previous + (1.0 - alpha) * current
    # Keep the categorical anchor and observed image-center from the current frame for reporting.
    smoothed[2] = current[2]
    smoothed[3] = current[3]
    if len(smoothed) >= 5:
        smoothed[4] = current[4]
    return smoothed


def auto_project_point(
    point: List[float],
    bounds: np.ndarray,
    pitch_window: np.ndarray,
    pitch_length: float,
    pitch_width: float,
) -> List[float]:
    left, right, top, bottom = [float(v) for v in bounds]
    pitch_center, window_length = float(pitch_window[0]), float(pitch_window[1])
    x_norm = (float(point[0]) - left) / max(right - left, 1e-6)
    y_norm = (float(point[1]) - top) / max(bottom - top, 1e-6)

    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    y_norm = float(np.clip(y_norm, 0.0, 1.0))

    # Expand the upper part of the image slightly to reduce perspective compression.
    depth = y_norm ** 0.85

    pitch_left = max(0.0, pitch_center - window_length / 2.0)
    pitch_right = min(pitch_length, pitch_center + window_length / 2.0)
    if pitch_right <= pitch_left:
        pitch_left, pitch_right = 0.0, pitch_length

    pitch_x = round(pitch_left + x_norm * (pitch_right - pitch_left), 3)
    pitch_y = round(depth * pitch_width, 3)
    return [pitch_x, pitch_y]


def draw_pitch(
    pitch_length: float,
    pitch_width: float,
    scale: int = 10,
    margin: int = 40,
) -> np.ndarray:
    canvas_width = int(pitch_length * scale + margin * 2)
    canvas_height = int(pitch_width * scale + margin * 2)
    image = np.full((canvas_height, canvas_width, 3), (42, 110, 57), dtype=np.uint8)
    white = (245, 245, 245)
    top_left = (margin, margin)
    bottom_right = (margin + int(pitch_length * scale), margin + int(pitch_width * scale))

    cv2.rectangle(image, top_left, bottom_right, white, 2)

    mid_x = margin + int((pitch_length / 2.0) * scale)
    cv2.line(image, (mid_x, margin), (mid_x, bottom_right[1]), white, 2)

    center = (mid_x, margin + int((pitch_width / 2.0) * scale))
    cv2.circle(image, center, int(9.15 * scale), white, 2)
    cv2.circle(image, center, 3, white, -1)

    penalty_box_depth = int(16.5 * scale)
    penalty_box_width = int(40.3 * scale)
    six_yard_depth = int(5.5 * scale)
    six_yard_width = int(18.32 * scale)
    penalty_spot_offset = int(11.0 * scale)

    box_top = margin + int(((pitch_width - 40.3) / 2.0) * scale)
    box_bottom = box_top + penalty_box_width
    cv2.rectangle(image, (margin, box_top), (margin + penalty_box_depth, box_bottom), white, 2)
    cv2.rectangle(
        image,
        (bottom_right[0] - penalty_box_depth, box_top),
        (bottom_right[0], box_bottom),
        white,
        2,
    )

    six_top = margin + int(((pitch_width - 18.32) / 2.0) * scale)
    six_bottom = six_top + six_yard_width
    cv2.rectangle(image, (margin, six_top), (margin + six_yard_depth, six_bottom), white, 2)
    cv2.rectangle(
        image,
        (bottom_right[0] - six_yard_depth, six_top),
        (bottom_right[0], six_bottom),
        white,
        2,
    )

    cv2.circle(image, (margin + penalty_spot_offset, center[1]), 3, white, -1)
    cv2.circle(image, (bottom_right[0] - penalty_spot_offset, center[1]), 3, white, -1)
    return image


def draw_players_on_pitch(
    pitch_image: np.ndarray,
    detections: List[Dict[str, Any]],
    pitch_length: float,
    pitch_width: float,
    frame_name: str,
    scale: int = 10,
    margin: int = 40,
) -> np.ndarray:
    output = pitch_image.copy()
    colors = {
        "player_team_a": (230, 90, 70),
        "player_team_b": (70, 120, 230),
        "goalkeeper_team_a": (255, 196, 0),
        "goalkeeper_team_b": (0, 196, 255),
        "other": (220, 220, 220),
        "unknown": (220, 220, 220),
    }

    for detection in detections:
        pitch_point = detection.get("foot_point_pitch")
        if not pitch_point:
            continue
        x_px = margin + int((pitch_point[0] / pitch_length) * pitch_length * scale)
        y_px = margin + int((pitch_point[1] / pitch_width) * pitch_width * scale)
        color = colors.get(detection.get("label", "unknown"), colors["unknown"])
        cv2.circle(output, (x_px, y_px), 6, color, -1)

    cv2.putText(
        output,
        frame_name,
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return output


def map_detections(
    detections_path: Path,
    calibration_path: Path,
    output_path: Path,
    visualization_dir: Optional[Path],
    alpha: float,
    ransac_threshold: float,
    max_frames: Optional[int],
) -> Dict[str, Any]:
    detections_by_frame = load_json(detections_path)
    calibration_data = load_json(calibration_path)

    pitch_meta = calibration_data.get("pitch", {})
    pitch_length = float(pitch_meta.get("length", DEFAULT_PITCH_LENGTH))
    pitch_width = float(pitch_meta.get("width", DEFAULT_PITCH_WIDTH))

    if not 0.0 <= alpha < 1.0:
        raise ValueError("EMA alpha must be in the range [0, 1).")

    frame_names = sorted(detections_by_frame.keys())
    if max_frames is not None:
        frame_names = frame_names[:max_frames]

    results: Dict[str, Any] = {
        "meta": {
            "detections_source": str(detections_path),
            "calibration_source": str(calibration_path),
            "pitch": {
                "length": pitch_length,
                "width": pitch_width,
                "units": "meters",
            },
            "ema_alpha": alpha,
            "ransac_threshold": ransac_threshold,
        },
        "frames": {},
    }

    base_pitch = draw_pitch(pitch_length, pitch_width)
    previous_smoothed_h: Optional[np.ndarray] = None
    previous_raw_h: Optional[np.ndarray] = None

    if visualization_dir is not None:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    for frame_name in frame_names:
        frame_detections = detections_by_frame[frame_name]
        calibration_entry, calibration_source = get_frame_calibration(calibration_data, frame_name)

        raw_homography = None
        inlier_indices: List[int] = []
        reprojection_error = None
        calibration_status = "ok"

        if calibration_entry is not None:
            raw_homography, inlier_indices, reprojection_error = estimate_homography(
                calibration_entry["image_points"],
                calibration_entry["pitch_points"],
                ransac_threshold,
            )
            previous_raw_h = raw_homography
        elif previous_raw_h is not None:
            raw_homography = previous_raw_h
            calibration_source = "carry_forward"
            calibration_status = "reused_previous"
        else:
            calibration_source = "missing"
            calibration_status = "missing"

        smoothed_homography = None
        mapped_detections = []

        if raw_homography is not None:
            smoothed_homography = smooth_homography(previous_smoothed_h, raw_homography, alpha)
            previous_smoothed_h = smoothed_homography

        for detection in frame_detections:
            mapped = dict(detection)
            if smoothed_homography is not None:
                foot_point_pitch = project_point(detection["foot_point"], smoothed_homography)
                mapped["foot_point_pitch"] = foot_point_pitch
                mapped["inside_pitch"] = point_inside_pitch(foot_point_pitch, pitch_length, pitch_width)
            else:
                mapped["foot_point_pitch"] = None
                mapped["inside_pitch"] = False
            mapped_detections.append(mapped)

        results["frames"][frame_name] = {
            "calibration_source": calibration_source,
            "calibration_status": calibration_status,
            "num_detections": len(frame_detections),
            "num_homography_inliers": len(inlier_indices),
            "reprojection_error": reprojection_error,
            "homography": smoothed_homography.tolist() if smoothed_homography is not None else None,
            "players": mapped_detections,
        }

        if visualization_dir is not None:
            visualization = draw_players_on_pitch(
                base_pitch,
                mapped_detections,
                pitch_length,
                pitch_width,
                frame_name,
            )
            cv2.imwrite(str(visualization_dir / f"{Path(frame_name).stem}_pitch.png"), visualization)

    dump_json(output_path, results)
    return results


def auto_map_detections(
    detections_path: Path,
    output_path: Path,
    visualization_dir: Optional[Path],
    alpha: float,
    max_frames: Optional[int],
    images_dir: Optional[Path],
) -> Dict[str, Any]:
    detections_by_frame = load_json(detections_path)

    if not 0.0 <= alpha < 1.0:
        raise ValueError("EMA alpha must be in the range [0, 1).")

    pitch_length = DEFAULT_PITCH_LENGTH
    pitch_width = DEFAULT_PITCH_WIDTH
    frame_width, frame_height = infer_frame_size(detections_by_frame)
    resolved_images_dir = images_dir if images_dir is not None else infer_images_dir(detections_path)

    frame_names = sorted(detections_by_frame.keys())
    if max_frames is not None:
        frame_names = frame_names[:max_frames]

    results: Dict[str, Any] = {
        "meta": {
            "detections_source": str(detections_path),
            "mapping_mode": "auto_detection_driven",
            "pitch": {
                "length": pitch_length,
                "width": pitch_width,
                "units": "meters",
            },
            "frame_size": {
                "width": frame_width,
                "height": frame_height,
            },
            "images_source": str(resolved_images_dir) if resolved_images_dir is not None else None,
            "ema_alpha": alpha,
        },
        "frames": {},
    }

    base_pitch = draw_pitch(pitch_length, pitch_width)
    previous_bounds: Optional[np.ndarray] = None
    previous_window: Optional[np.ndarray] = None

    if visualization_dir is not None:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    for frame_name in frame_names:
        frame_detections = detections_by_frame[frame_name]
        current_bounds = estimate_auto_bounds(frame_detections, frame_width, frame_height)
        smoothed_bounds = smooth_bounds(previous_bounds, current_bounds, alpha)
        previous_bounds = smoothed_bounds
        image_path = resolved_images_dir / frame_name if resolved_images_dir is not None else None
        view_cues = detect_view_cues(image_path)
        current_window = estimate_auto_pitch_window(
            frame_detections,
            smoothed_bounds,
            frame_width,
            pitch_length,
            previous_window,
            view_cues,
        )
        smoothed_window = smooth_pitch_window(previous_window, current_window, alpha)
        previous_window = smoothed_window

        view_region = "middle"
        anchor_ratio = float(smoothed_window[2])
        if anchor_ratio < 0.4:
            view_region = "left"
        elif anchor_ratio > 0.6:
            view_region = "right"

        mapped_detections = []
        for detection in frame_detections:
            mapped = dict(detection)
            foot_point_pitch = auto_project_point(
                detection["foot_point"],
                smoothed_bounds,
                smoothed_window,
                pitch_length,
                pitch_width,
            )
            mapped["foot_point_pitch"] = foot_point_pitch
            mapped["inside_pitch"] = point_inside_pitch(foot_point_pitch, pitch_length, pitch_width)
            mapped_detections.append(mapped)

        results["frames"][frame_name] = {
            "calibration_source": "auto_detection_driven",
            "calibration_status": "heuristic",
            "num_detections": len(frame_detections),
            "auto_bounds_image": [round(float(v), 3) for v in smoothed_bounds.tolist()],
            "auto_pitch_window": {
                "center_x": round(float(smoothed_window[0]), 3),
                "visible_length": round(float(smoothed_window[1]), 3),
                "region": view_region,
                "image_center_ratio": round(float(smoothed_window[3]), 3),
            },
            "view_cues": view_cues,
            "players": mapped_detections,
        }

        if visualization_dir is not None:
            visualization = draw_players_on_pitch(
                base_pitch,
                mapped_detections,
                pitch_length,
                pitch_width,
                frame_name,
            )
            cv2.imwrite(str(visualization_dir / f"{Path(frame_name).stem}_pitch.png"), visualization)

    dump_json(output_path, results)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Person 3 pipeline: map person-2 foot points into canonical 2D pitch coordinates."
    )
    parser.add_argument(
        "--mode",
        choices=["homography", "auto"],
        default="homography",
        help="Use calibrated homography mapping or automatic detection-driven heuristic mapping.",
    )
    parser.add_argument(
        "--detections",
        required=True,
        type=Path,
        help="Path to person 2 detection JSON, for example mapping_data/SNGS-060_detections.json.",
    )
    parser.add_argument(
        "--calibration",
        required=False,
        type=Path,
        help="Path to calibration JSON with image-to-pitch point correspondences.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON for mapped player positions.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="Optional directory to save top-down pitch visualizations per frame.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Optional frame directory for automatic view cues. If omitted, the script tries to infer img1.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.8,
        help="Temporal smoothing strength for homography EMA. Higher means smoother.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=5.0,
        help="OpenCV RANSAC reprojection threshold in image pixels.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for quick experiments.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "auto":
        results = auto_map_detections(
            detections_path=args.detections,
            output_path=args.output,
            visualization_dir=args.visualization_dir,
            alpha=args.ema_alpha,
            max_frames=args.max_frames,
            images_dir=args.images_dir,
        )
    else:
        if args.calibration is None:
            parser.error("--calibration is required when --mode homography is used.")
        results = map_detections(
            detections_path=args.detections,
            calibration_path=args.calibration,
            output_path=args.output,
            visualization_dir=args.visualization_dir,
            alpha=args.ema_alpha,
            ransac_threshold=args.ransac_threshold,
            max_frames=args.max_frames,
        )

    print(f"Mapped {len(results['frames'])} frames.")
    print(f"Saved mapped detections to {args.output}")
    if args.visualization_dir is not None:
        print(f"Saved pitch visualizations to {args.visualization_dir}")


if __name__ == "__main__":
    main()
