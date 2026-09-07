#!/usr/bin/env python3
"""Build a quick AVM asset with the currently selected fisheye correction.

The generated AVMAP still consumes raw 1920x1080 frames. For every BEV output
pixel it first applies the inverse homography in the corrected-image plane and
then samples the supplied FMAP to obtain a raw-camera coordinate. Consequently
the Jetson performs one runtime remap, not an undistort pass followed by AVM.
"""

import argparse
import hashlib
import importlib.util
import json
import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("front", "left", "right", "bottom")
AVMAP_HEADER = struct.Struct("<8s7I4xQ4Q")

# Parameters from fish-eye-undistort.zip. K_NEW is the exact rectified camera
# matrix used by the supplied root-level undistort_map.bin.
K_CURRENT = np.asarray(
    [
        [516.8013980933, 0.0, 959.9479673357],
        [0.0, 517.0228242052, 542.0684276335],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
D_CURRENT = np.asarray([0.25, -0.0025, -0.0025, 0.002], dtype=np.float64)
K_NEW_CURRENT = np.asarray(
    [
        [455.175203582352, 0.0, 960.0],
        [0.0, 455.175203582352, 540.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_interface(root: Path):
    path = root / "03_scripts" / "avm_4stream_bottom_rear_interface.py"
    spec = importlib.util.spec_from_file_location("avm_quick_recipe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_fmap(path: Path):
    with path.open("rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"truncated FMAP header: {path}")
        magic, version, width, height = struct.unpack("<4sIII", header)
        if magic != b"FMAP" or version != 1:
            raise ValueError(f"unsupported FMAP header: magic={magic!r}, version={version}")
        count = width * height
        map_x = np.fromfile(stream, dtype="<f4", count=count)
        map_y = np.fromfile(stream, dtype="<f4", count=count)
        if map_x.size != count or map_y.size != count or stream.read(1):
            raise ValueError(f"FMAP payload size mismatch: {path}")
    return width, height, map_x.reshape(height, width), map_y.reshape(height, width)


def point_rows(points, camera):
    details = points["point_details"][camera]
    return list(details.values())


def fit_homography(points, camera):
    rows = point_rows(points, camera)
    raw = np.asarray([row["raw_xy"] for row in rows], dtype=np.float64).reshape(-1, 1, 2)
    target = np.asarray([row["target_bev_xy"] for row in rows], dtype=np.float64)
    corrected = cv2.fisheye.undistortPoints(
        raw, K_CURRENT, D_CURRENT.reshape(4, 1), R=np.eye(3), P=K_NEW_CURRENT
    ).reshape(-1, 2)
    homography, _mask = cv2.findHomography(corrected, target, method=0)
    if homography is None:
        raise RuntimeError(f"could not fit homography for {camera}")
    homography /= homography[2, 2]
    projected = cv2.perspectiveTransform(
        corrected.astype(np.float32).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - target, axis=1)
    report = {
        "point_count": len(rows),
        "mean_px": float(np.mean(errors)),
        "max_px": float(np.max(errors)),
        "p95_px": float(np.percentile(errors, 95)),
        "homography_corrected_to_bev": homography.tolist(),
    }
    return homography, report


def expand_spec(spec, width, height):
    map_x = np.full((height, width), -1.0, np.float32)
    map_y = np.full((height, width), -1.0, np.float32)
    if spec is None or spec.get("bbox") is None:
        return map_x, map_y
    x0, y0, x1, y1 = spec["bbox"]
    valid = spec["valid"]
    roi_x = map_x[y0:y1, x0:x1]
    roi_y = map_y[y0:y1, x0:x1]
    roi_x[valid] = spec["map_x"][valid]
    roi_y[valid] = spec["map_y"][valid]
    return map_x, map_y


def bgr_to_limited_yuv(bgr):
    blue = bgr[..., 0].astype(np.float32)
    green = bgr[..., 1].astype(np.float32)
    red = bgr[..., 2].astype(np.float32)
    y = 16.0 + (65.738 * red + 129.057 * green + 25.064 * blue) / 256.0
    u = 128.0 + (-37.945 * red - 74.494 * green + 112.439 * blue) / 256.0
    v = 128.0 + (112.439 * red - 94.154 * green - 18.285 * blue) / 256.0
    return np.stack([y, u, v], axis=-1).clip(0, 255).astype(np.uint8)


def write_asset(output, stitcher):
    width, height = stitcher.output_size
    maps_x = []
    maps_y = []
    for camera in CAMERAS[:3]:
        map_x, map_y = expand_spec(stitcher._formal_specs[camera], width, height)
        maps_x.append(map_x)
        maps_y.append(map_y)

    projector = stitcher._bottom_singleH_projector
    bottom_x = np.full((height, width), -1.0, np.float32)
    bottom_y = np.full((height, width), -1.0, np.float32)
    if projector.bbox is not None:
        x0, y0, x1, y1 = projector.bbox
        bottom_x[y0:y1, x0:x1] = projector.map_x
        bottom_y[y0:y1, x0:x1] = projector.map_y
    maps_x.append(bottom_x)
    maps_y.append(bottom_y)

    maps_x = np.stack(maps_x).astype("<f4")
    maps_y = np.stack(maps_y).astype("<f4")
    weights = np.stack([stitcher.weights[camera].astype(np.float32) for camera in CAMERAS])
    valid = (maps_x >= 0.0) & (maps_y >= 0.0)
    weights *= valid.astype(np.float32)
    total = weights.sum(axis=0)
    nonzero = total > 1e-8
    weights[:, nonzero] /= total[nonzero]
    weights[:, ~nonzero] = 0.0

    overlay = np.zeros((height, width, 4), np.uint8)
    if getattr(stitcher, "_static_overlay_enabled", False):
        x0, y0, x1, y1 = stitcher._static_overlay_bbox
        overlay[y0:y1, x0:x1, :3] = bgr_to_limited_yuv(stitcher._static_overlay_rgb_roi)
        overlay[y0:y1, x0:x1, 3] = stitcher._static_overlay_alpha_roi
    flags = 1 if overlay[..., 3].any() else 0

    header = AVMAP_HEADER.pack(
        b"AVMAP01\0",
        1,
        stitcher.raw_size[0],
        stitcher.raw_size[1],
        width,
        height,
        4,
        flags,
        width * height,
        0,
        0,
        0,
        0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(header)
        maps_x.tofile(stream)
        maps_y.tofile(stream)
        weights.astype("<f4").tofile(stream)
        if flags:
            overlay.tofile(stream)

    return maps_x, maps_y, weights, overlay, flags


def verify_asset(path):
    with path.open("rb") as stream:
        raw = stream.read(AVMAP_HEADER.size)
    if len(raw) != AVMAP_HEADER.size:
        raise ValueError("truncated AVMAP header")
    values = AVMAP_HEADER.unpack(raw)
    magic, version, src_w, src_h, width, height, cameras, flags, pixels = values[:9]
    if magic != b"AVMAP01\0" or version != 1 or cameras != 4 or pixels != width * height:
        raise ValueError(f"invalid AVMAP header: {values}")
    expected = AVMAP_HEADER.size + 3 * cameras * pixels * 4
    if flags & 1:
        expected += pixels * 4
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"AVMAP size mismatch: expected={expected}, actual={actual}")
    return {
        "magic": magic.decode("ascii").rstrip("\0"),
        "version": version,
        "source_size": [src_w, src_h],
        "canvas_size": [width, height],
        "camera_count": cameras,
        "flags": flags,
        "pixel_count": pixels,
        "file_size": actual,
    }


def compare_with_baseline(path, maps_x, maps_y):
    with path.open("rb") as stream:
        values = AVMAP_HEADER.unpack(stream.read(AVMAP_HEADER.size))
    _magic, _version, _sw, _sh, width, height, cameras, _flags, pixels = values[:9]
    if cameras != maps_x.shape[0] or (height, width) != maps_x.shape[1:]:
        raise ValueError("baseline AVMAP dimensions do not match quick asset")
    baseline_x = np.memmap(
        path, dtype="<f4", mode="r", offset=AVMAP_HEADER.size, shape=(cameras, height, width)
    )
    baseline_y = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=AVMAP_HEADER.size + cameras * pixels * 4,
        shape=(cameras, height, width),
    )
    result = {}
    for index, camera in enumerate(CAMERAS):
        valid = (
            (maps_x[index] >= 0.0)
            & (maps_y[index] >= 0.0)
            & (baseline_x[index] >= 0.0)
            & (baseline_y[index] >= 0.0)
        )
        delta = np.hypot(
            maps_x[index][valid] - baseline_x[index][valid],
            maps_y[index][valid] - baseline_y[index][valid],
        )
        result[camera] = {
            "common_valid_pixels": int(valid.sum()),
            "mean_source_shift_px": float(np.mean(delta)) if delta.size else None,
            "median_source_shift_px": float(np.median(delta)) if delta.size else None,
            "p95_source_shift_px": float(np.percentile(delta, 95)) if delta.size else None,
            "max_source_shift_px": float(np.max(delta)) if delta.size else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suanfa_dir", type=Path)
    parser.add_argument("fmap", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()

    root = args.suanfa_dir.resolve()
    fmap_path = args.fmap.resolve()
    output = args.output.resolve()
    report_path = (args.report or output.with_suffix(".report.json")).resolve()
    config_path = root / "00_configs" / "loader_avm_bev_config_zhuangzaiji2.json"
    points_path = root / "01_point_files" / "loader_avm_homography_points_raw_zhuangzaiji2.json"
    bottom_points_path = (
        root / "01_point_files" / "loader_avm_homography_points_raw_zhuangzaiji2_bottom_frame340.json"
    )

    fmap_w, fmap_h, fmap_x, fmap_y = load_fmap(fmap_path)
    if (fmap_w, fmap_h) != (1920, 1080):
        raise ValueError(f"quick exporter requires a 1920x1080 FMAP, got {fmap_w}x{fmap_h}")

    reference_x, reference_y = cv2.fisheye.initUndistortRectifyMap(
        K_CURRENT,
        D_CURRENT.reshape(4, 1),
        np.eye(3),
        K_NEW_CURRENT,
        (fmap_w, fmap_h),
        cv2.CV_32FC1,
    )
    fmap_error_x = fmap_x - reference_x
    fmap_error_y = fmap_y - reference_y
    fmap_max_error = float(max(np.max(np.abs(fmap_error_x)), np.max(np.abs(fmap_error_y))))
    if fmap_max_error > 1e-3:
        raise ValueError(
            f"FMAP does not match the configured current K/D/Knew; max difference={fmap_max_error:.6f}px"
        )

    points = load_json(points_path)
    bottom_points = load_json(bottom_points_path)
    homographies = {}
    fit_report = {}
    for camera in ("front", "left", "right", "back"):
        homographies[camera], fit_report[camera] = fit_homography(points, camera)
    homographies["bottom"], fit_report["bottom"] = fit_homography(bottom_points, "bottom")

    module = load_interface(root)
    bottom_h = homographies["bottom"]

    class QuickStitcher(module.ROIOptimizedAVMBottomRearStitcher):
        def __init__(self, *stitcher_args, **stitcher_kwargs):
            self._quick_bottom_h = bottom_h
            super().__init__(*stitcher_args, **stitcher_kwargs)

        def _load_homographies(self):
            single = self._quick_bottom_h
            self.H_board3 = self.translation @ single
            self.H_board4 = self.translation @ single
            self.H_center = self.translation @ single
            self.center_source_polygon = module.load_json(
                self.matrix_dir / "bottom_center_source_polygon.json"
            )["source_polygon_xy"]
            target_polygon = module.load_json(self.matrix_dir / "bottom_center_target_polygon.json")[
                "target_polygon_xy"
            ]
            self.center_target_polygon = [
                [
                    int(round(x + module.SAFE_CANVAS["dx"])),
                    int(round(y + module.SAFE_CANVAS["dy"])),
                ]
                for x, y in target_polygon
            ]
            self.bottom_singleH_meta = {
                "path": "generated in memory by export_avm_asset_quick.py",
                "key": "current_undistort_quick_H",
                "uses_fisheye_undistort": True,
                "raw_points_directly_used_for_h": False,
                "reprojection_error": fit_report["bottom"],
            }

    matrix_payload = {
        "project_matrices": {
            camera: homographies[camera].tolist() for camera in ("front", "left", "right", "back")
        }
    }
    with tempfile.TemporaryDirectory(prefix="avm_quick_") as temp_dir:
        matrix_path = Path(temp_dir) / "project_matrices_quick.json"
        matrix_path.write_text(json.dumps(matrix_payload), encoding="utf-8")
        stitcher = QuickStitcher(config_path, matrix_path, use_cuda=False)

        # Replace the recipe's legacy balance=0 maps with the exact current FMAP,
        # then rebuild all composite BEV-to-raw maps.
        stitcher._undistort_map_x = fmap_x
        stitcher._undistort_map_y = fmap_y
        stitcher._prepare_fast_specs()
        stitcher._prepare_bottom_singleH_composite_map()

        maps_x, maps_y, weights, overlay, flags = write_asset(output, stitcher)

    asset_info = verify_asset(output)
    report = {
        "asset": asset_info,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "fmap": {
            "path": str(fmap_path),
            "sha256": sha256_file(fmap_path),
            "size": [fmap_w, fmap_h],
            "opencv_reference_max_error_px": fmap_max_error,
            "opencv_reference_rms_error_px": float(
                np.sqrt(np.mean(fmap_error_x * fmap_error_x + fmap_error_y * fmap_error_y))
            ),
        },
        "calibration": {
            "model": "opencv_fisheye_4_parameter",
            "K": K_CURRENT.tolist(),
            "D": D_CURRENT.tolist(),
            "Knew": K_NEW_CURRENT.tolist(),
            "shared_by_four_avm_cameras": True,
        },
        "homography_fit": fit_report,
        "valid_pixels": {
            camera: int(((maps_x[index] >= 0.0) & (maps_y[index] >= 0.0)).sum())
            for index, camera in enumerate(CAMERAS)
        },
        "weight_sum": {
            "nonzero_pixels": int((weights.sum(axis=0) > 1e-8).sum()),
            "max_normalization_error": float(
                np.max(np.abs(weights.sum(axis=0)[weights.sum(axis=0) > 1e-8] - 1.0))
            ),
        },
        "overlay_enabled": bool(flags & 1),
        "overlay_nonzero_alpha_pixels": int((overlay[..., 3] > 0).sum()),
    }
    if args.baseline:
        baseline = args.baseline.resolve()
        report["baseline"] = {
            "path": str(baseline),
            "sha256": sha256_file(baseline),
            "map_shift": compare_with_baseline(baseline, maps_x, maps_y),
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"sha256={report['output_sha256']}")
    print(f"source={asset_info['source_size']}, canvas={asset_info['canvas_size']}, overlay={bool(flags)}")
    print(f"report={report_path}")
    for camera in CAMERAS:
        metrics = fit_report[camera]
        print(
            f"{camera}: points={metrics['point_count']}, mean={metrics['mean_px']:.3f}px, "
            f"p95={metrics['p95_px']:.3f}px, max={metrics['max_px']:.3f}px"
        )


if __name__ == "__main__":
    main()
