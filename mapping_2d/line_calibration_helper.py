import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


INTERSECTION_LANDMARKS: Dict[str, Dict[str, object]] = {
    "left_box_top_corner": {
        "pitch": [16.5, 13.85],
        "lines": ["left_penalty_box_vertical", "penalty_box_top"],
    },
    "left_box_bottom_corner": {
        "pitch": [16.5, 54.15],
        "lines": ["left_penalty_box_vertical", "penalty_box_bottom"],
    },
    "right_box_top_corner": {
        "pitch": [88.5, 13.85],
        "lines": ["right_penalty_box_vertical", "penalty_box_top"],
    },
    "right_box_bottom_corner": {
        "pitch": [88.5, 54.15],
        "lines": ["right_penalty_box_vertical", "penalty_box_bottom"],
    },
    "left_six_yard_top_corner": {
        "pitch": [5.5, 24.84],
        "lines": ["left_six_yard_vertical", "six_yard_top"],
    },
    "left_six_yard_bottom_corner": {
        "pitch": [5.5, 43.16],
        "lines": ["left_six_yard_vertical", "six_yard_bottom"],
    },
    "right_six_yard_top_corner": {
        "pitch": [99.5, 24.84],
        "lines": ["right_six_yard_vertical", "six_yard_top"],
    },
    "right_six_yard_bottom_corner": {
        "pitch": [99.5, 43.16],
        "lines": ["right_six_yard_vertical", "six_yard_bottom"],
    },
    "center_line_top_touchline": {
        "pitch": [52.5, 0.0],
        "lines": ["center_line", "top_touchline"],
    },
    "center_line_bottom_touchline": {
        "pitch": [52.5, 68.0],
        "lines": ["center_line", "bottom_touchline"],
    },
    "left_top_corner": {
        "pitch": [0.0, 0.0],
        "lines": ["left_goal_line", "top_touchline"],
    },
    "left_bottom_corner": {
        "pitch": [0.0, 68.0],
        "lines": ["left_goal_line", "bottom_touchline"],
    },
    "right_top_corner": {
        "pitch": [105.0, 0.0],
        "lines": ["right_goal_line", "top_touchline"],
    },
    "right_bottom_corner": {
        "pitch": [105.0, 68.0],
        "lines": ["right_goal_line", "bottom_touchline"],
    },
}


DEFAULT_LANDMARK_ORDER = [
    "left_box_top_corner",
    "left_box_bottom_corner",
    "center_line_top_touchline",
    "center_line_bottom_touchline",
]


def load_json_if_exists(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def ensure_base_schema(payload: Dict) -> Dict:
    if "pitch" not in payload:
        payload["pitch"] = {
            "length": PITCH_LENGTH,
            "width": PITCH_WIDTH,
            "units": "meters",
        }
    if "frames" not in payload:
        payload["frames"] = {}
    return payload


def parse_landmark_names(raw_value: str) -> List[str]:
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    invalid = [name for name in names if name not in INTERSECTION_LANDMARKS]
    if invalid:
        raise ValueError(
            f"Unknown landmarks: {', '.join(invalid)}. "
            f"Valid names are: {', '.join(sorted(INTERSECTION_LANDMARKS))}"
        )
    return names


def compute_intersection(
    a1: Tuple[int, int],
    a2: Tuple[int, int],
    b1: Tuple[int, int],
    b2: Tuple[int, int],
) -> Optional[Tuple[float, float]]:
    p1 = np.array([a1[0], a1[1], 1.0], dtype=np.float64)
    p2 = np.array([a2[0], a2[1], 1.0], dtype=np.float64)
    p3 = np.array([b1[0], b1[1], 1.0], dtype=np.float64)
    p4 = np.array([b2[0], b2[1], 1.0], dtype=np.float64)

    line_a = np.cross(p1, p2)
    line_b = np.cross(p3, p4)
    intersection = np.cross(line_a, line_b)

    if abs(intersection[2]) < 1e-8:
        return None
    return (float(intersection[0] / intersection[2]), float(intersection[1] / intersection[2]))


def clip_line_to_image(
    line_pt1: Tuple[int, int],
    line_pt2: Tuple[int, int],
    image_shape: Tuple[int, int, int],
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    height, width = image_shape[:2]
    x1, y1 = line_pt1
    x2, y2 = line_pt2
    dx = x2 - x1
    dy = y2 - y1
    intersections: List[Tuple[int, int]] = []

    if dx != 0:
        for x in (0, width - 1):
            t = (x - x1) / dx
            y = y1 + t * dy
            if 0 <= y <= height - 1:
                intersections.append((int(round(x)), int(round(y))))

    if dy != 0:
        for y in (0, height - 1):
            t = (y - y1) / dy
            x = x1 + t * dx
            if 0 <= x <= width - 1:
                intersections.append((int(round(x)), int(round(y))))

    unique_points: List[Tuple[int, int]] = []
    for point in intersections:
        if point not in unique_points:
            unique_points.append(point)

    if len(unique_points) < 2:
        return None
    return unique_points[0], unique_points[1]


def current_prompt(landmark_names: List[str], clicks: List[Tuple[int, int]]) -> str:
    landmark_index = len(clicks) // 4
    click_stage = len(clicks) % 4
    if landmark_index >= len(landmark_names):
        return "All landmarks collected. Press s to save."

    landmark_name = landmark_names[landmark_index]
    line_names = INTERSECTION_LANDMARKS[landmark_name]["lines"]
    prompts = [
        f"{landmark_name}: click point 1 on {line_names[0]}",
        f"{landmark_name}: click point 2 on {line_names[0]}",
        f"{landmark_name}: click point 1 on {line_names[1]}",
        f"{landmark_name}: click point 2 on {line_names[1]}",
    ]
    return prompts[click_stage]


def annotate_image(
    image: np.ndarray,
    landmark_names: List[str],
    clicks: List[Tuple[int, int]],
) -> np.ndarray:
    canvas = image.copy()
    overlay_lines = [
        "Left click: add line point",
        "u: undo last click",
        "s: save calibration",
        "q: quit without saving",
    ]
    for index, line in enumerate(overlay_lines):
        cv2.putText(
            canvas,
            line,
            (20, 30 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    colors = ((0, 180, 255), (255, 180, 0))

    for landmark_index, landmark_name in enumerate(landmark_names):
        start = landmark_index * 4
        subset = clicks[start:start + 4]

        for idx, point in enumerate(subset):
            color = colors[0] if idx < 2 else colors[1]
            cv2.circle(canvas, point, 5, color, -1)
            if idx in (1, 3):
                pair_start = subset[idx - 1]
                clipped = clip_line_to_image(pair_start, point, canvas.shape)
                if clipped is not None:
                    cv2.line(canvas, clipped[0], clipped[1], color, 2)

        if len(subset) == 4:
            intersection = compute_intersection(subset[0], subset[1], subset[2], subset[3])
            if intersection is not None:
                ix, iy = int(round(intersection[0])), int(round(intersection[1]))
                cv2.circle(canvas, (ix, iy), 7, (0, 0, 255), -1)
                cv2.putText(
                    canvas,
                    landmark_name,
                    (ix + 8, iy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    canvas,
                    f"{landmark_name}: parallel lines",
                    (20, 150 + landmark_index * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

    status = current_prompt(landmark_names, clicks)
    cv2.putText(
        canvas,
        status,
        (20, canvas.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def build_frame_entry(
    landmark_names: List[str],
    clicks: List[Tuple[int, int]],
) -> Dict:
    landmarks = {}
    image_points = []
    pitch_points = []

    for landmark_index, landmark_name in enumerate(landmark_names):
        start = landmark_index * 4
        subset = clicks[start:start + 4]
        if len(subset) != 4:
            raise ValueError(f"Landmark {landmark_name} is incomplete.")

        intersection = compute_intersection(subset[0], subset[1], subset[2], subset[3])
        if intersection is None:
            raise ValueError(f"Landmark {landmark_name} has parallel or invalid lines.")

        pitch_point = INTERSECTION_LANDMARKS[landmark_name]["pitch"]
        line_names = INTERSECTION_LANDMARKS[landmark_name]["lines"]
        image_xy = [round(intersection[0], 3), round(intersection[1], 3)]

        landmarks[landmark_name] = {
            "image": image_xy,
            "pitch": pitch_point,
            "lines": {
                str(line_names[0]): [
                    [float(subset[0][0]), float(subset[0][1])],
                    [float(subset[1][0]), float(subset[1][1])],
                ],
                str(line_names[1]): [
                    [float(subset[2][0]), float(subset[2][1])],
                    [float(subset[3][0]), float(subset[3][1])],
                ],
            },
        }
        image_points.append(image_xy)
        pitch_points.append(pitch_point)

    return {
        "landmarks": landmarks,
        "image_points": image_points,
        "pitch_points": pitch_points,
    }


def collect_line_clicks(image: np.ndarray, landmark_names: List[str]) -> List[Tuple[int, int]]:
    window_name = "line calibration helper"
    clicks: List[Tuple[int, int]] = []
    total_required = len(landmark_names) * 4

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < total_required:
            clicks.append((x, y))
            print(f"Added click {len(clicks)}/{total_required}: ({x}, {y})")
            print(current_prompt(landmark_names, clicks))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("Landmarks to calibrate via line intersections:")
    for index, landmark_name in enumerate(landmark_names, start=1):
        info = INTERSECTION_LANDMARKS[landmark_name]
        print(
            f"{index}. {landmark_name} -> pitch {info['pitch']} "
            f"using lines {info['lines'][0]} and {info['lines'][1]}"
        )
    print(current_prompt(landmark_names, clicks))

    while True:
        annotated = annotate_image(image, landmark_names, clicks)
        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("u"):
            if clicks:
                removed = clicks.pop()
                print(f"Removed last click {removed}")
                print(current_prompt(landmark_names, clicks))
        elif key == ord("s"):
            if len(clicks) != total_required:
                print(f"Need {total_required} clicks before saving.")
                continue
            try:
                build_frame_entry(landmark_names, clicks)
            except ValueError as exc:
                print(exc)
                continue
            break
        elif key == ord("q"):
            clicks = []
            break

    cv2.destroyAllWindows()
    return clicks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create calibration correspondences from line intersections."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to a broadcast frame image.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Calibration JSON to create or update.",
    )
    parser.add_argument(
        "--frame-name",
        default=None,
        help="Frame key to store in JSON. Defaults to the image filename.",
    )
    parser.add_argument(
        "--save-as",
        choices=["frame", "default"],
        default="frame",
        help="Save this calibration under frames[frame_name] or as the shared default entry.",
    )
    parser.add_argument(
        "--landmarks",
        default=",".join(DEFAULT_LANDMARK_ORDER),
        help="Comma-separated intersection landmarks to collect.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    landmark_names = parse_landmark_names(args.landmarks)

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    clicks = collect_line_clicks(image, landmark_names)
    if not clicks:
        print("Calibration cancelled. Nothing was saved.")
        return

    payload = ensure_base_schema(load_json_if_exists(args.output))
    entry = build_frame_entry(landmark_names, clicks)

    if args.save_as == "default":
        payload["default"] = entry
        save_target = "default"
    else:
        frame_name = args.frame_name or args.image.name
        payload["frames"][frame_name] = entry
        save_target = f"frames[{frame_name}]"

    save_json(args.output, payload)
    print(f"Saved calibration to {args.output} at {save_target}")


if __name__ == "__main__":
    main()
