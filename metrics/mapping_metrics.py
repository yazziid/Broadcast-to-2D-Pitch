import json
from typing import Dict, Any, List

import numpy as np


def load_json(path: str) -> Dict[str, Any]:
    """
    Load a JSON file from disk.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(values: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for a list of numeric values.
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compute_spatial_spread(points: List[List[float]]) -> float:
    """
    Estimate how spatially dispersed the projected player positions are on the pitch.

    This is computed as the mean distance of projected points from their centroid.
    Larger values indicate a more spread-out distribution of projected players.
    """
    if len(points) < 2:
        return 0.0

    arr = np.asarray(points, dtype=np.float32)
    center = np.mean(arr, axis=0)
    distances = np.linalg.norm(arr - center, axis=1)
    return float(np.mean(distances))


def evaluate_mapping(mapping_json_path: str) -> None:
    """
    Evaluate the 2-D mapping results produced by the project.

    The script extracts frame-level and summary-level metrics directly from the
    saved mapping JSON, including:
    - reprojection error
    - homography inlier ratio
    - inside/outside pitch ratios
    - average detection confidence
    - projected spatial spread
    - average projected position on the canonical pitch
    """
    data = load_json(mapping_json_path)
    frames = data["frames"]

    reprojection_errors = []
    inlier_ratios = []
    inside_ratios = []
    outside_ratios = []
    avg_confidences = []
    spatial_spreads = []
    avg_pitch_x = []
    avg_pitch_y = []

    print("\n2-D Mapping Metrics")

    for frame_name, frame_data in frames.items():
        detections = int(frame_data.get("num_detections", 0))
        inliers = int(frame_data.get("num_homography_inliers", 0))
        reprojection_error = float(frame_data.get("reprojection_error", 0.0))
        players = frame_data.get("players", [])

        inside_players = [p for p in players if p.get("inside_pitch", False)]
        outside_players = [p for p in players if not p.get("inside_pitch", False)]

        inside_count = len(inside_players)
        outside_count = len(outside_players)

        inside_ratio = inside_count / detections if detections > 0 else 0.0
        outside_ratio = outside_count / detections if detections > 0 else 0.0
        inlier_ratio = inliers / detections if detections > 0 else 0.0

        confidences = [float(p.get("confidence", 0.0)) for p in players]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        projected_points = [p["foot_point_pitch"] for p in players if "foot_point_pitch" in p]
        spread = compute_spatial_spread(projected_points)

        mean_x = float(np.mean([p[0] for p in projected_points])) if projected_points else 0.0
        mean_y = float(np.mean([p[1] for p in projected_points])) if projected_points else 0.0

        reprojection_errors.append(reprojection_error)
        inlier_ratios.append(inlier_ratio)
        inside_ratios.append(inside_ratio)
        outside_ratios.append(outside_ratio)
        avg_confidences.append(avg_conf)
        spatial_spreads.append(spread)
        avg_pitch_x.append(mean_x)
        avg_pitch_y.append(mean_y)

        """ print(f"\n{frame_name}")
        print(f"Detections:                    {detections}")
        print(f"Homography inliers:            {inliers}")
        print(f"Inlier ratio:                  {inlier_ratio:.3f}")
        print(f"Reprojection error:            {reprojection_error:.3f}")
        print(f"Inside pitch count:            {inside_count}")
        print(f"Outside pitch count:           {outside_count}")
        print(f"Inside pitch ratio:            {inside_ratio:.3f}")
        print(f"Outside pitch ratio:           {outside_ratio:.3f}")
        print(f"Average confidence:            {avg_conf:.3f}")
        print(f"Projected spatial spread:      {spread:.3f}")
        print(f"Mean projected X:              {mean_x:.3f}")
        print(f"Mean projected Y:              {mean_y:.3f}") """

    print("\nSummary Statistics for 2d mapping")
    print(f"Avg reprojection error:        {summarize(reprojection_errors)['mean']:.3f}")
    print(f"Std reprojection error:        {summarize(reprojection_errors)['std']:.3f}")
    print(f"Avg inlier ratio:              {summarize(inlier_ratios)['mean']:.3f}")
    print(f"Avg inside pitch ratio:        {summarize(inside_ratios)['mean']:.3f}")
    print(f"Avg outside pitch ratio:       {summarize(outside_ratios)['mean']:.3f}")
    print(f"Avg player confidence:         {summarize(avg_confidences)['mean']:.3f}")
    print(f"Avg projected spatial spread:  {summarize(spatial_spreads)['mean']:.3f}")
    print(f"Avg projected X:               {summarize(avg_pitch_x)['mean']:.3f}")
    print(f"Avg projected Y:               {summarize(avg_pitch_y)['mean']:.3f}")


if __name__ == "__main__":
    # Mapping output generated by the 2-D projection pipeline.
    MAPPING_JSON = "../mapping_data/SNGS-128_pitch_positions_3frames.json"
    
    evaluate_mapping(MAPPING_JSON)