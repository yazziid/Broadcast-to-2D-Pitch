import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


PITCH_LANDMARKS: Dict[str, List[float]] = {
    "left_top_corner": [0.0, 0.0],
    "left_bottom_corner": [0.0, 68.0],
    "right_top_corner": [105.0, 0.0],
    "right_bottom_corner": [105.0, 68.0],
    "center_spot": [52.5, 34.0],
    "center_circle_left": [43.35, 34.0],
    "center_circle_right": [61.65, 34.0],
    "center_circle_top": [52.5, 24.85],
    "center_circle_bottom": [52.5, 43.15],
    "center_line_top_touchline": [52.5, 0.0],
    "center_line_bottom_touchline": [52.5, 68.0],
    "left_penalty_spot": [11.0, 34.0],
    "right_penalty_spot": [94.0, 34.0],
    "left_box_top_corner": [16.5, 13.85],
    "left_box_bottom_corner": [16.5, 54.15],
    "right_box_top_corner": [88.5, 13.85],
    "right_box_bottom_corner": [88.5, 54.15],
    "left_six_yard_top_corner": [5.5, 24.84],
    "left_six_yard_bottom_corner": [5.5, 43.16],
    "right_six_yard_top_corner": [99.5, 24.84],
    "right_six_yard_bottom_corner": [99.5, 43.16],
}


DEFAULT_LANDMARK_ORDER = [
    "left_box_top_corner",
    "left_box_bottom_corner",
    "center_spot",
    "right_box_top_corner",
    "right_box_bottom_corner",
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


def parse_landmark_names(raw_value: str) -> List[str]:
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    invalid = [name for name in names if name not in PITCH_LANDMARKS]
    if invalid:
        raise ValueError(
            f"Unknown landmarks: {', '.join(invalid)}. "
            f"Valid names are: {', '.join(sorted(PITCH_LANDMARKS))}"
        )
    return names


def annotate_image(
    image: np.ndarray,
    landmark_names: List[str],
    clicked_points: List[Tuple[int, int]],
) -> np.ndarray:
    canvas = image.copy()
    overlay_lines = [
        "Left click: add point",
        "u: undo last point",
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

    for index, point in enumerate(clicked_points):
        name = landmark_names[index]
        cv2.circle(canvas, point, 6, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            name,
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if len(clicked_points) < len(landmark_names):
        next_name = landmark_names[len(clicked_points)]
        status = f"Next landmark: {next_name}"
    else:
        status = "All landmarks collected. Press s to save."

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
    clicked_points: List[Tuple[int, int]],
) -> Dict:
    landmarks = {}
    image_points = []
    pitch_points = []

    for name, image_point in zip(landmark_names, clicked_points):
        pitch_point = PITCH_LANDMARKS[name]
        image_xy = [float(image_point[0]), float(image_point[1])]
        landmarks[name] = {
            "image": image_xy,
            "pitch": pitch_point,
        }
        image_points.append(image_xy)
        pitch_points.append(pitch_point)

    return {
        "landmarks": landmarks,
        "image_points": image_points,
        "pitch_points": pitch_points,
    }


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


def collect_points(image: np.ndarray, landmark_names: List[str]) -> List[Tuple[int, int]]:
    window_name = "person3 calibration helper"
    clicked_points: List[Tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < len(landmark_names):
            clicked_points.append((x, y))
            print(f"Added {landmark_names[len(clicked_points) - 1]} -> ({x}, {y})")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("Landmarks to click in order:")
    for index, name in enumerate(landmark_names, start=1):
        print(f"{index}. {name} -> pitch {PITCH_LANDMARKS[name]}")

    while True:
        annotated = annotate_image(image, landmark_names, clicked_points)
        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("u"):
            if clicked_points:
                removed = landmark_names[len(clicked_points) - 1]
                clicked_points.pop()
                print(f"Removed last point for {removed}")
        elif key == ord("s"):
            if len(clicked_points) < 4:
                print("Need at least 4 points before saving.")
                continue
            if len(clicked_points) != len(landmark_names):
                print("Not all requested landmarks are collected yet.")
                continue
            break
        elif key == ord("q"):
            clicked_points = []
            break

    cv2.destroyAllWindows()
    return clicked_points


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive helper to create person-3 calibration correspondences."
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
        help="Comma-separated landmark names to click in order.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    landmark_names = parse_landmark_names(args.landmarks)

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    clicked_points = collect_points(image, landmark_names)
    if not clicked_points:
        print("Calibration cancelled. Nothing was saved.")
        return

    payload = ensure_base_schema(load_json_if_exists(args.output))
    entry = build_frame_entry(landmark_names, clicked_points)

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
