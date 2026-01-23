#!/usr/bin/env python3
"""
Visualize a point cloud stored as a .npy file.

Supports:
- Nx3: XYZ
- Nx6: XYZRGB (RGB can be 0-1 or 0-255)
- dict saved via np.save(..., allow_pickle=True) with keys like 'xyz', 'points', 'rgb', 'colors'
"""

import argparse
import numpy as np

def _extract_xyz_rgb(arr):
    # If someone saved a dict as npy (pickle)
    if isinstance(arr, dict):
        # common key variants
        xyz = None
        for k in ["xyz", "points", "pts", "pcd", "pointcloud", "pos", "positions"]:
            if k in arr:
                xyz = np.asarray(arr[k])
                break
        if xyz is None:
            raise ValueError(f"Dict .npy but no known XYZ key found. Keys: {list(arr.keys())}")

        rgb = None
        for k in ["rgb", "color", "colors", "colours"]:
            if k in arr:
                rgb = np.asarray(arr[k])
                break

        return xyz, rgb

    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (Nx3 or Nx6). Got shape {arr.shape}")

    if arr.shape[1] == 3:
        return arr[:, :3], None
    if arr.shape[1] >= 6:
        return arr[:, :3], arr[:, 3:6]

    raise ValueError(f"Expected Nx3 or Nx6+. Got shape {arr.shape}")

def _normalize_rgb(rgb):
    rgb = rgb.astype(np.float32)
    # If any value > 1, assume 0-255
    if np.nanmax(rgb) > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", type=str, help="Path to pointcloud .npy")
    parser.add_argument("--every", type=int, default=1, help="Downsample by taking every k-th point")
    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size (0 disables)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale XYZ by this factor")
    args = parser.parse_args()

    try:
        import open3d as o3d
    except ImportError as e:
        raise SystemExit(
            "Open3D not installed. Install with:\n"
            "  pip install open3d\n"
            "or use the Matplotlib fallback script."
        ) from e

    data = np.load(args.npy_path, allow_pickle=True)
    # If pickled dict/obj, numpy returns 0-d array of dtype=object
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        data = data.item()

    xyz, rgb = _extract_xyz_rgb(data)

    # basic clean + optional scale/downsample
    xyz = np.asarray(xyz, dtype=np.float32) * float(args.scale)
    if args.every > 1:
        xyz = xyz[::args.every]
        if rgb is not None:
            rgb = rgb[::args.every]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(_normalize_rgb(rgb))

    if args.voxel and args.voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel))

    o3d.visualization.draw_geometries([pcd], window_name="NPY Point Cloud Viewer")

if __name__ == "__main__":
    main()
