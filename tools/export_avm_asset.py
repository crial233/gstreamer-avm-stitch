#!/usr/bin/env python3
"""Export the supplied Python AVM recipe into a deterministic AVMAP v1 file.

Run this on a machine that has numpy and opencv-python installed:
  python3 tools/export_avm_asset.py /opt/avm/suanfa /opt/calibration/avm_zhuangzaiji2.bin
"""
import argparse
import importlib.util
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ("front", "left", "right", "bottom")


def load_interface(root: Path):
    path = root / "03_scripts" / "avm_4stream_bottom_rear_interface.py"
    spec = importlib.util.spec_from_file_location("avm_recipe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def expand_spec(spec, width, height):
    mx = np.full((height, width), -1.0, np.float32)
    my = np.full((height, width), -1.0, np.float32)
    if spec is None or spec.get("bbox") is None:
        return mx, my
    x0, y0, x1, y1 = spec["bbox"]
    valid = spec["valid"]
    rx, ry = mx[y0:y1, x0:x1], my[y0:y1, x0:x1]
    rx[valid] = spec["map_x"][valid]
    ry[valid] = spec["map_y"][valid]
    return mx, my


def bgr_to_limited_yuv(bgr):
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    y = 16.0 + (65.738 * r + 129.057 * g + 25.064 * b) / 256.0
    u = 128.0 + (-37.945 * r - 74.494 * g + 112.439 * b) / 256.0
    v = 128.0 + (112.439 * r - 94.154 * g - 18.285 * b) / 256.0
    return np.stack([y, u, v], axis=-1).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suanfa_dir", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    root = args.suanfa_dir.resolve()
    mod = load_interface(root)
    stitcher = mod.ROIOptimizedAVMBottomRearStitcher(
        root / "00_configs" / "loader_avm_bev_config_zhuangzaiji2.json",
        root / "02_matrices" / "loader_avm_project_matrices_from_raw_zhuangzaiji2.json",
        use_cuda=False,
    )
    width, height = stitcher.output_size
    maps_x, maps_y = [], []
    for camera in CAMERAS[:3]:
        x, y = expand_spec(stitcher._formal_specs[camera], width, height)
        maps_x.append(x); maps_y.append(y)
    projector = stitcher._bottom_singleH_projector
    bx = np.full((height, width), -1.0, np.float32)
    by = np.full((height, width), -1.0, np.float32)
    if projector.bbox is not None:
        x0, y0, x1, y1 = projector.bbox
        bx[y0:y1, x0:x1] = projector.map_x
        by[y0:y1, x0:x1] = projector.map_y
    maps_x.append(bx); maps_y.append(by)
    weights = np.stack([stitcher.weights[c].astype(np.float32) for c in CAMERAS])
    valid = np.stack([(m >= 0) for m in maps_x]) & np.stack([(m >= 0) for m in maps_y])
    weights *= valid.astype(np.float32)
    total = weights.sum(axis=0)
    nz = total > 1e-8
    weights[:, nz] /= total[nz]
    weights[:, ~nz] = 0.0

    overlay = np.zeros((height, width, 4), np.uint8)
    if getattr(stitcher, "_static_overlay_enabled", False):
        x0, y0, x1, y1 = stitcher._static_overlay_bbox
        overlay[y0:y1, x0:x1, :3] = bgr_to_limited_yuv(stitcher._static_overlay_rgb_roi)
        overlay[y0:y1, x0:x1, 3] = stitcher._static_overlay_alpha_roi
    flags = 1 if overlay[..., 3].any() else 0
    header = struct.pack(
        "<8s7I4xQ4Q", b"AVMAP01\0", 1, stitcher.raw_size[0], stitcher.raw_size[1],
        width, height, 4, flags, width * height, 0, 0, 0, 0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(header)
        np.stack(maps_x).astype("<f4").tofile(f)
        np.stack(maps_y).astype("<f4").tofile(f)
        weights.astype("<f4").tofile(f)
        if flags:
            overlay.tofile(f)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"source={stitcher.raw_size}, canvas={stitcher.output_size}, overlay={bool(flags)}")


if __name__ == "__main__":
    main()
