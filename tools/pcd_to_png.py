"""
pcd_to_png.py
-------------
Render a single .pcd file to a .png image using the same Open3D
camera-setup and render pipeline used in the evaluation script.

Usage:
    python pcd_to_png.py input.pcd output.png [--size 512] [--point_size 3.0]
"""

import argparse
import os

import numpy as np
import open3d as o3d


# ─────────────────────────────────────────────
# Camera helpers  (copied verbatim from eval script)
# ─────────────────────────────────────────────

def _make_camera_params_from_gt(gt_pcd, fov_deg=60.0, width=512, height=512,
                                 point_size=3.5, visible=False):
    bbox   = gt_pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = float(np.linalg.norm(extent) * 0.5)
    radius = radius if radius > 0 else 1.0

    z  = 2 * radius
    xy = np.array([1.0, 0.0], dtype=np.float64) * radius
    th = np.deg2rad(45.0)
    Rz = np.array([[np.cos(th), -np.sin(th)],
                   [np.sin(th),  np.cos(th)]], dtype=np.float64)
    xy = Rz.dot(xy)

    cam_pos = center + np.array([xy[0], xy[1], z], dtype=np.float64)
    cam_up  = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=visible)
    vis.add_geometry(gt_pcd)
    ctr   = vis.get_view_control()
    front = (center - cam_pos).astype(np.float64)
    front /= (np.linalg.norm(front) + 1e-12)
    ctr.set_lookat(center.tolist())
    ctr.set_front(front.tolist())
    ctr.set_up(cam_up.tolist())
    ctr.set_zoom(0.75)
    params = ctr.convert_to_pinhole_camera_parameters()

    fov_rad = np.deg2rad(float(fov_deg))
    fx = fy = 0.5 * width / np.tan(0.5 * fov_rad)
    intr = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, width * 0.5, height * 0.5
    )
    params.intrinsic        = intr
    params.intrinsic.width  = int(width)
    params.intrinsic.height = int(height)
    vis.destroy_window()

    return params


def _render_with_camera_params(pcd, params, width=512, height=512,
                                point_size=3.5, visible=False):
    win_w = int(getattr(params.intrinsic, "width",  width))
    win_h = int(getattr(params.intrinsic, "height", height))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=win_w, height=win_h, visible=visible)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    try:
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    except Exception:
        pass

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size       = float(point_size)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────

def render_pcd_to_png(pcd_path: str, out_path: str,
                      size: int = 512, point_size: float = 3.0,
                      color: tuple = (0.2, 0.6, 1.0)) -> str:
    """
    Load a .pcd file and render it to a PNG.

    Parameters
    ----------
    pcd_path   : path to input .pcd
    out_path   : path for output .png
    size       : square render resolution (pixels)
    point_size : Open3D point size
    color      : uniform RGB color applied if the cloud has no colours
                 (values in [0, 1])

    Returns
    -------
    out_path on success.
    """
    assert os.path.exists(pcd_path), f"PCD not found: {pcd_path}"

    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    assert pts.shape[0] > 0, "Point cloud is empty."

    # Apply a uniform colour if the cloud has none
    if not pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.array(color, dtype=np.float64), (pts.shape[0], 1))
        )

    # Build camera from the cloud itself (treat it as its own "GT")
    params = _make_camera_params_from_gt(
        pcd, fov_deg=60.0, width=size, height=size,
        point_size=point_size, visible=False,
    )

    # Render
    img_np = _render_with_camera_params(
        pcd, params, width=size, height=size,
        point_size=point_size, visible=False,
    )

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    from PIL import Image
    Image.fromarray(img_np).save(out_path)
    print(f"Saved {out_path}  ({size}x{size}, {pts.shape[0]} points)")
    return out_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a .pcd file to a .png image.")
    parser.add_argument("pcd_path",   type=str,   help="Input .pcd file")
    parser.add_argument("out_path",   type=str,   help="Output .png file")
    parser.add_argument("--size",       type=int,   default=512,
                        help="Square render size in pixels (default: 512)")
    parser.add_argument("--point_size", type=float, default=3.0,
                        help="Open3D point size (default: 3.0)")
    parser.add_argument("--color",      type=float, nargs=3,
                        default=[0.2, 0.6, 1.0],
                        metavar=("R", "G", "B"),
                        help="Fallback uniform RGB colour if cloud has none (default: 0.2 0.6 1.0)")
    args = parser.parse_args()

    render_pcd_to_png(
        pcd_path   = args.pcd_path,
        out_path   = args.out_path,
        size       = args.size,
        point_size = args.point_size,
        color      = tuple(args.color),
    )