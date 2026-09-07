#!/usr/bin/env python3
"""Simulate articulated-loader bucket trajectories on a corrected camera image.

The script uses the calibration embedded in
calibration/avm_zhuangzaiji2_quick.report.json.  It is an offline visual test;
no CAN/GStreamer input is required.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "calibration" / "avm_zhuangzaiji2_quick.report.json"
CAMERAS = ("front", "left", "right", "back")

# Dimensions from 转弯轨迹线-装载机_v20220908.xlsx, converted mm -> cm.
FRONT_AXLE_FROM_HINGE_CM = 166.0
REAR_AXLE_FROM_HINGE_CM = 179.0
BUCKET_AHEAD_OF_FRONT_AXLE_CM = 300.0
BUCKET_HALF_WIDTH_CM = 297.6 / 2.0
MAX_ARTICULATION_DEG = 38.0

# AVMAP quick asset canvas: x is right, y is forward, 1 pixel == 1 cm.
BEV_X_MIN_CM = -800.0
BEV_Y_MAX_CM = 850.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay simulated loader bucket trajectories on a camera frame."
    )
    parser.add_argument("input", type=Path, help="1920x1080 input image")
    parser.add_argument("-o", "--output", type=Path, default=Path("trajectory_overlay.png"))
    parser.add_argument("--camera", choices=CAMERAS, default="front")
    parser.add_argument(
        "--angle", type=float, default=20.0,
        help="front/rear frame articulation angle in degrees, [-38, 38]",
    )
    parser.add_argument(
        "--raw-fisheye", action="store_true",
        help="undistort the input first; omit when the image is already corrected",
    )
    parser.add_argument("--length", type=float, default=800.0, help="trajectory length in cm")
    parser.add_argument("--line-width", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.88)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    # imdecode handles Chinese paths reliably on Windows.
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"cannot encode output as {suffix}")
    encoded.tofile(path)


def undistort_fisheye(image: np.ndarray, calibration: dict) -> np.ndarray:
    h, w = image.shape[:2]
    expected = tuple(calibration["asset"]["source_size"])
    if (w, h) != expected:
        raise ValueError(f"raw image must be {expected[0]}x{expected[1]}, got {w}x{h}")
    c = calibration["calibration"]
    k = np.asarray(c["K"], dtype=np.float64)
    d = np.asarray(c["D"], dtype=np.float64).reshape(4, 1)
    knew = np.asarray(c["Knew"], dtype=np.float64)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        k, d, np.eye(3), knew, (w, h), cv2.CV_32FC1
    )
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def rotate_about(points: np.ndarray, center: np.ndarray, radians: np.ndarray) -> np.ndarray:
    relative = points - center
    cosine = np.cos(radians)
    sine = np.sin(radians)
    x = center[0] + cosine * relative[0] - sine * relative[1]
    y = center[1] + sine * relative[0] + cosine * relative[1]
    return np.column_stack((x, y))


def bucket_trajectories(angle_deg: float, length_cm: float, samples: int = 320) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right bucket-corner paths in vehicle ground coordinates.

    The hinge is the origin. At zero articulation both frame headings are +Y.
    Positive articulation rotates the front frame toward +X (vehicle right).
    Constant articulation and no tyre slip are assumed.
    """
    gamma = math.radians(angle_deg)
    heading = np.array([math.sin(gamma), math.cos(gamma)], dtype=np.float64)
    right = np.array([math.cos(gamma), -math.sin(gamma)], dtype=np.float64)
    front_axle = FRONT_AXLE_FROM_HINGE_CM * heading
    bucket_center = front_axle + BUCKET_AHEAD_OF_FRONT_AXLE_CM * heading
    left_start = bucket_center - BUCKET_HALF_WIDTH_CM * right
    right_start = bucket_center + BUCKET_HALF_WIDTH_CM * right

    distances = np.linspace(0.0, max(1.0, length_cm), samples)
    if abs(gamma) < math.radians(0.05):
        delta = distances[:, None] * heading
        return left_start + delta, right_start + delta

    # Intersection of the normals through front and rear axle centres.
    mu = (FRONT_AXLE_FROM_HINGE_CM * math.cos(gamma) + REAR_AXLE_FROM_HINGE_CM) / math.sin(gamma)
    icr = front_axle + mu * right
    front_radius = float(np.linalg.norm(front_axle - icr))

    # Pick rotation sign whose front-axle tangent points along the front heading.
    radial = front_axle - icr
    ccw_tangent = np.array([-radial[1], radial[0]])
    rotation_sign = 1.0 if float(np.dot(ccw_tangent, heading)) >= 0.0 else -1.0
    swept = rotation_sign * distances / front_radius
    return rotate_about(left_start, icr, swept), rotate_about(right_start, icr, swept)


def ground_to_corrected(points_cm: np.ndarray, corrected_to_bev: np.ndarray) -> np.ndarray:
    bev = np.column_stack(
        (points_cm[:, 0] - BEV_X_MIN_CM, BEV_Y_MAX_CM - points_cm[:, 1])
    ).astype(np.float32)
    bev_to_corrected = np.linalg.inv(corrected_to_bev)
    return cv2.perspectiveTransform(bev.reshape(-1, 1, 2), bev_to_corrected).reshape(-1, 2)


def draw_clipped_polyline(
    canvas: np.ndarray, points: np.ndarray, color: tuple[int, int, int], width: int
) -> None:
    h, w = canvas.shape[:2]
    finite = np.isfinite(points).all(axis=1)
    inside = finite & (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    start = None
    for i, valid in enumerate(inside):
        if valid and start is None:
            start = i
        if start is not None and (not valid or i == len(inside) - 1):
            end = i + 1 if valid else i
            if end - start >= 2:
                segment = np.rint(points[start:end]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [segment], False, color, width, cv2.LINE_AA)
            start = None


def main() -> int:
    args = parse_args()
    if not -MAX_ARTICULATION_DEG <= args.angle <= MAX_ARTICULATION_DEG:
        raise ValueError(f"--angle must be within +/-{MAX_ARTICULATION_DEG} degrees")
    if args.length <= 0 or args.line_width <= 0 or not 0.0 <= args.alpha <= 1.0:
        raise ValueError("length/line-width must be positive and alpha must be within [0, 1]")

    calibration = json.loads(args.report.read_text(encoding="utf-8"))
    image = read_image(args.input)
    if args.raw_fisheye:
        image = undistort_fisheye(image, calibration)
    expected = tuple(calibration["asset"]["source_size"])
    actual = (image.shape[1], image.shape[0])
    if actual != expected:
        # WeChat/exported screenshots can be a top-left-aligned crop of the
        # corrected frame.  Keeping the original pixel origin makes the same
        # homography valid; projected content below the crop is simply clipped.
        if not args.raw_fisheye and actual[0] == expected[0] and actual[1] < expected[1]:
            print(
                f"warning: treating {actual[0]}x{actual[1]} as a top-left crop "
                f"of the calibrated {expected[0]}x{expected[1]} frame"
            )
        else:
            raise ValueError(f"corrected image must be {expected[0]}x{expected[1]}")

    left_ground, right_ground = bucket_trajectories(args.angle, args.length)
    homography = np.asarray(
        calibration["homography_fit"][args.camera]["homography_corrected_to_bev"],
        dtype=np.float64,
    )
    left_image = ground_to_corrected(left_ground, homography)
    right_image = ground_to_corrected(right_ground, homography)

    overlay = image.copy()
    draw_clipped_polyline(overlay, left_image, (0, 255, 0), args.line_width)
    draw_clipped_polyline(overlay, right_image, (0, 180, 255), args.line_width)
    result = cv2.addWeighted(overlay, args.alpha, image, 1.0 - args.alpha, 0.0)
    cv2.putText(
        result, f"articulation: {args.angle:+.1f} deg  camera: {args.camera}",
        (36, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
    )
    write_image(args.output, result)
    print(f"saved: {args.output.resolve()}")
    if args.show:
        cv2.imshow("trajectory overlay", result)
        cv2.waitKey(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
