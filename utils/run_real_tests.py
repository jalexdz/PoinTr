import argparse
import os
from collections import defaultdict
import numpy as np
import torch
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from tools import builder
from utils.config import cfg_from_yaml_file
from tools.predictor import AdaPoinTrPredictor
from utils.metrics import Metrics
from datasets.io import IO

# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot real-world evaluation with ICP heuristic.
#
# GT is sampled from the provided mesh files (Poisson disk, n=8192),
# matching exactly the sim pipeline.  The same GT cloud is reused for every
# view of a given asset.
#
# Per (asset, view) produces a combined N-ablation-row grid:
#
#   ┌────────────┬──────────┬──────────┬──────────┬──────────┐
#   │  Title                                                  │
#   ├────────────┼──────────┼──────────┼──────────┼──────────┤
#   │  Baseline  │  Part.   │  Pred.   │ KNN Err. │GT+ICP Ov.│
#   ├────────────┼──────────┼──────────┼──────────┼──────────┤
#   │  Ablation1 │  ...     │  ...     │  ...     │  ...     │
#   └────────────┴──────────┴──────────┴──────────┴──────────┘
#
# Outputs:
#   <out_path>/grids/combined/<asset>/view_XXXX.pdf   — ablation-row grids
#   <out_path>/grids/individual/<abl>/<asset>/...pdf  — one 2×2 per ablation
#   <out_path>/plots/strips/*.pdf                     — strip plots
#   <out_path>/combined_results.csv
#   <out_path>/latex_table.tex
# ─────────────────────────────────────────────────────────────────────────────

RCPARAMS = {
    "font.size":        13,
    "axes.labelsize":   11,
    "axes.titlesize":   13,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
}


# ─────────────────────────────────────────────
# GT sampling
# ─────────────────────────────────────────────

def build_gt_dict(mesh_specs: list[tuple[str, str]],
                  n_points: int = 8192,
                  seed: int = 0) -> dict:
    """
    Sample GT point clouds from meshes once upfront.

    Args:
        mesh_specs : [(taxonomy_id, mesh_path), ...]
        n_points   : Poisson disk sample count — match your sim eval (8192)
        seed       : for reproducibility

    Returns:
        {taxonomy_id: (N, 3) float32 np.ndarray}
    """
    np.random.default_rng(seed)   # not used directly but sets global state
    gt_dict = {}
    for tid, mesh_path in mesh_specs:
        print(f"  Sampling GT for '{tid}' from {mesh_path} ...")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        if mesh.is_empty():
            raise RuntimeError(f"Failed to load mesh: {mesh_path}")
        gt_pts = np.asarray(
            mesh.sample_points_poisson_disk(n_points).points,
            dtype=np.float32,
        )
        gt_dict[tid] = gt_pts
        print(f"    {gt_pts.shape[0]} points sampled.")
    return gt_dict


# ─────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────

def _norm_from_partial(partial: np.ndarray):
    centroid     = np.mean(partial, axis=0)
    partial_norm = (partial - centroid) / 2.0
    return partial_norm.astype(np.float32), centroid


# ─────────────────────────────────────────────
# ICP
# ─────────────────────────────────────────────

def _fpfh_features(pcd, voxel_size: float):
    """Compute FPFH features for global registration."""
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )


def run_icp(source_pts: np.ndarray, target_pts: np.ndarray,
            max_correspondence_dist: float = 0.05,
            max_iter: int = 100) -> dict:
    """
    Two-stage registration:
      1. RANSAC global registration using FPFH features — handles arbitrary
         rotational offsets between the real-world completion and the mesh GT.
      2. Point-to-point ICP refinement starting from the RANSAC result.

    Both clouds are normalised to unit scale before registration so the
    voxel/distance parameters are scale-invariant, then the result is mapped
    back to the original scale.

    Returns:
        registered_pts  — (N,3) transformed source (original scale)
        fitness         — ICP inlier fraction [0,1]
        inlier_rmse_mm  — ICP inlier RMSE (same units as inputs)
        transformation  — (4,4) homogeneous transform (original scale)
    """
    # ── Normalise both clouds to unit bounding-box scale ─────────────────────
    # This makes voxel sizes and distance thresholds scale-invariant.
    all_pts  = np.concatenate([source_pts, target_pts], axis=0)
    extent   = all_pts.max(axis=0) - all_pts.min(axis=0)
    scale    = float(np.linalg.norm(extent))
    scale    = scale if scale > 1e-9 else 1.0

    src_norm = source_pts.astype(np.float64) / scale
    tgt_norm = target_pts.astype(np.float64) / scale

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_norm)

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_norm)

    # ── Voxel downsample for RANSAC (speed) ───────────────────────────────────
    voxel = 0.05   # 5% of bounding-box diagonal in normalised space
    src_down = src_pcd.voxel_down_sample(voxel)
    tgt_down = tgt_pcd.voxel_down_sample(voxel)

    src_fpfh = _fpfh_features(src_down, voxel)
    tgt_fpfh = _fpfh_features(tgt_down, voxel)

    # ── RANSAC global registration ─────────────────────────────────────────────
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    T_ransac = ransac.transformation  # in normalised space

    # ── ICP refinement from RANSAC result ─────────────────────────────────────
    icp_dist_norm = max_correspondence_dist / scale   # convert to normalised space
    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=icp_dist_norm,
        init=T_ransac,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )

    # ── Apply transform and convert back to original scale ────────────────────
    # The rotation/translation of the normalised transform is the same rotation
    # in world space; only the translation needs to be rescaled.
    T_norm = result.transformation.copy()
    T_world = T_norm.copy()
    T_world[:3, 3] *= scale   # rescale translation only; rotation is unchanged

    src_world = o3d.geometry.PointCloud()
    src_world.points = o3d.utility.Vector3dVector(source_pts.astype(np.float64))
    src_world.transform(T_world)

    return {
        "registered_pts":  np.asarray(src_world.points).astype(np.float32),
        "fitness":         float(result.fitness),
        "inlier_rmse_mm":  float(result.inlier_rmse * scale),   # back to original units
        "transformation":  T_world,
    }


def compute_icp_metrics(registered_pts: np.ndarray,
                         gt_pts: np.ndarray) -> dict:
    """CD and F1 between ICP-registered completion and GT (same call as sim eval)."""
    reg_t = torch.from_numpy(registered_pts).float().cuda().unsqueeze(0)
    gt_t  = torch.from_numpy(gt_pts).float().cuda().unsqueeze(0)
    m     = Metrics.get(reg_t, gt_t, require_emd=False)
    return {
        "icp_cd_mm": float(2 * m[1]),
        "icp_f1":    float(m[0]),
    }


# ─────────────────────────────────────────────
# Camera + rendering
# ─────────────────────────────────────────────

def _make_camera_params(pcd, fov_deg=60.0, width=512, height=512, visible=False):
    bbox   = pcd.get_axis_aligned_bounding_box()
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
    vis.add_geometry(pcd)
    ctr   = vis.get_view_control()
    front = (center - cam_pos).astype(np.float64)
    front /= np.linalg.norm(front) + 1e-12
    ctr.set_lookat(center.tolist())
    ctr.set_front(front.tolist())
    ctr.set_up(cam_up.tolist())
    ctr.set_zoom(0.75)
    params = ctr.convert_to_pinhole_camera_parameters()

    fov_rad = np.deg2rad(float(fov_deg))
    fx = fy  = 0.5 * width / np.tan(0.5 * fov_rad)
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy,
                                              width * 0.5, height * 0.5)
    params.intrinsic        = intr
    params.intrinsic.width  = int(width)
    params.intrinsic.height = int(height)
    vis.destroy_window()
    return params


def _render(pcd, params, width=512, height=512, point_size=3.5, visible=False):
    win_w = int(getattr(params.intrinsic, "width", width))
    win_h = int(getattr(params.intrinsic, "height", height))
    vis   = o3d.visualization.Visualizer()
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


def _knn_error_colormap(pred_pts_raw, gt_pts_raw, cmap="turbo", vmax_mm=100.0):
    """
    GT points coloured by nearest-neighbour distance to predicted cloud.
    Distances in mm (x1000 from metre-scale inputs).
    Clouds are centroid-aligned before distance computation, matching sim eval.
    vmax_mm=100 is fixed so colours are comparable across views and ablations.
    """
    pred_pts = pred_pts_raw.copy().astype(np.float64)
    gt_pts   = gt_pts_raw.copy().astype(np.float64)
    pred_pts -= pred_pts.mean(axis=0)
    gt_pts   -= gt_pts.mean(axis=0)

    if pred_pts.size == 0 or gt_pts.size == 0:
        return np.zeros((gt_pts_raw.shape[0], 3)), np.zeros(gt_pts_raw.shape[0])

    pred_pcd_err = o3d.geometry.PointCloud()
    pred_pcd_err.points = o3d.utility.Vector3dVector(pred_pts)
    tree  = o3d.geometry.KDTreeFlann(pred_pcd_err)
    dists = np.zeros(gt_pts.shape[0], dtype=np.float32)
    for i, p in enumerate(gt_pts):
        _, _, d2 = tree.search_knn_vector_3d(p, 1)
        dists[i] = 1000.0 * np.sqrt(d2[0]) if len(d2) > 0 else 0.0  # metres -> mm
    norm   = np.clip(dists / float(vmax_mm), 0.0, 1.0)
    colors = plt.get_cmap(cmap)(norm)[:, :3]
    return colors.astype(np.float64), dists


# ─────────────────────────────────────────────
# Panel builder — 4 panels for one (view, ablation)
# ─────────────────────────────────────────────

def _build_4_panels(partial_pts, complete_pts, gt_pts, registered_pts,
                    cam_params, panel_w, panel_h, point_size):
    """
    Returns four PIL Images matching sim eval colour conventions:
        [0] Part.      — partial cloud  (red   [1,0,0])
        [1] Pred.      — raw completion (green [0,1,0])
        [2] KNN Err.   — GT coloured by dist to raw completion
                         (turbo, fixed vmax=100mm, centroid-aligned)
        [3] GT        — GT (blue [0,0,1]) + ICP-registered completion (green [0,1,0])
                         rendered with camera anchored on the overlay bbox so
                         orientation matches Pred.
    """
    part_pcd = o3d.geometry.PointCloud()
    part_pcd.points = o3d.utility.Vector3dVector(partial_pts.astype(np.float64))
    part_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.0, 0.0], (partial_pts.shape[0], 1)))   # red

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(complete_pts.astype(np.float64))
    pred_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 0.0], (complete_pts.shape[0], 1)))  # green

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pts.astype(np.float64))
    gt_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 0.0, 1.0], (gt_pts.shape[0], 1)))        # blue

    # KNN error: centroid-aligned, mm distances, fixed vmax=100mm, turbo
    err_colors, err_dists = _knn_error_colormap(complete_pts, gt_pts,
                                                cmap="turbo", vmax_mm=100.0)
    knn_pcd = o3d.geometry.PointCloud()
    knn_pcd.points = o3d.utility.Vector3dVector(gt_pts.astype(np.float64))
    knn_pcd.colors = o3d.utility.Vector3dVector(err_colors)

    # ICP overlay: GT blue + registered completion green
    reg_pcd = o3d.geometry.PointCloud()
    reg_pcd.points = o3d.utility.Vector3dVector(registered_pts.astype(np.float64))
    reg_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 0.0], (registered_pts.shape[0], 1)))  # green

    overlay_pcd = gt_pcd + reg_pcd

    # Translate the overlay to match the partial/pred coordinate frame so the
    # same camera shows everything right-side up and in-frame.
    # Strategy: shift overlay centroid to match the completion centroid.
    overlay_pts = np.concatenate([gt_pts, registered_pts], axis=0)
    overlay_shift = complete_pts.mean(axis=0) - overlay_pts.mean(axis=0)
    shifted_overlay = o3d.geometry.PointCloud()
    shifted_overlay.points = o3d.utility.Vector3dVector(
        (overlay_pts + overlay_shift).astype(np.float64))
    # Split colours: first gt_pts.shape[0] are GT (blue), rest are reg (green)
    n_gt  = gt_pts.shape[0]
    n_reg = registered_pts.shape[0]
    overlay_colors = np.vstack([
        np.tile([0.0, 0.0, 1.0], (n_gt,  1)),
        np.tile([0.0, 1.0, 0.0], (n_reg, 1)),
    ])
    shifted_overlay.colors = o3d.utility.Vector3dVector(overlay_colors)

    imgs = []
    for pcd in [part_pcd, pred_pcd, knn_pcd, shifted_overlay]:
        arr = _render(pcd, cam_params,
                      width=panel_w, height=panel_h, point_size=point_size)
        imgs.append(Image.fromarray(arr))

    err_mean = float(err_dists.mean())
    err_max  = float(err_dists.max())
    return imgs, err_mean, err_max


# ─────────────────────────────────────────────
# Fonts
# ─────────────────────────────────────────────

def _load_fonts():
    try:
        bold  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        reg   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        bold = reg = small = ImageFont.load_default()
    return bold, reg, small


# ─────────────────────────────────────────────
# 2×2 individual grid
# ─────────────────────────────────────────────

def compose_2x2_grid(panels, title_txt, panel_w, panel_h,
                     caption_height=30, footer_padding=30,
                     err_mean=None, err_max=None):
    """
    Identical layout to render_triplet_from_pcds.
    Fonts: title=20pt bold, captions=15pt. Title height dynamic.
    Grid = 2w x 2h, no inter-panel padding.
      top-left: Part. | top-right: Pred.
      bot-left: Err.  | bot-right: GT
    """
    try:
        title_font   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        caption_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        title_font = caption_font = ImageFont.load_default()

    err_label = (f"Err. (mean={err_mean:.2f} mm, max={err_max:.2f} mm)"
                 if err_mean is not None else "KNN Err.")
    cap_labels = ["Part.", "Pred.", err_label, "GT"]

    tmp_draw      = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bbox_title    = tmp_draw.textbbox((0, 0), title_txt, font=title_font)
    title_text_h  = bbox_title[3] - bbox_title[1]
    title_pad_top = 4
    title_height  = title_text_h + title_pad_top + 6

    pad_between = 0
    w, h    = panel_w, panel_h
    final_w = 2 * w + pad_between
    final_h = title_height + 2 * h + pad_between + caption_height + footer_padding

    out_image = Image.new("RGB", (final_w, final_h), (255, 255, 255))
    draw      = ImageDraw.Draw(out_image)
    y_panels  = title_height

    out_image.paste(panels[0], (0, y_panels))
    out_image.paste(panels[1], (w, y_panels))
    out_image.paste(panels[2], (0, y_panels + h))
    out_image.paste(panels[3], (w, y_panels + h))

    bbox    = draw.textbbox((0, 0), title_txt, font=title_font)
    title_x = max(0, (final_w - (bbox[2] - bbox[0])) // 2)
    draw.text((title_x, title_pad_top), title_txt, fill=(0, 0, 0), font=title_font)

    for i, label in enumerate(cap_labels):
        row_i, col_i = i // 2, i % 2
        cell_x       = col_i * (w + pad_between)
        cell_y_top   = y_panels + row_i * (h + pad_between)
        bbox_c       = draw.textbbox((0, 0), label, font=caption_font)
        cap_w        = bbox_c[2] - bbox_c[0]
        cap_h        = bbox_c[3] - bbox_c[1]
        cap_x        = cell_x + (w - cap_w) // 2
        cap_y        = cell_y_top + h + max(2, (caption_height - cap_h) // 2)
        draw.text((cap_x, cap_y), label, fill=(0, 0, 0), font=caption_font)

    return out_image

def compose_ablation_row_grid(rows_data, asset, view_id,
                               panel_w=384, panel_h=384,
                               row_label_w=110, caption_h=28,
                               row_gap=5, title_pad=36, footer=10):
    """
    rows_data: list of dicts per ablation:
        abl_name, panels (list of 4 PIL Images), metrics dict

    Layout:
        ┌────────────┬──────────┬──────────┬──────────┬──────────┐
        │  Title                                                  │
        ├────────────┼──────────┼──────────┼──────────┼──────────┤
        │  Baseline  │  Part.   │  Pred.   │ KNN Err. │GT+ICP Ov.│
        │  CD/F1/fit │                                            │
        ├────────────┼──────────┼──────────┼──────────┼──────────┤
        │  Ablation1 │  ...                                       │
        └────────────┴──────────┴──────────┴──────────┴──────────┘
        │            │  Part.   │  Pred.   │ KNN Err. │GT+ICP Ov.│ ← captions
    """
    bold_font, reg_font, small_font = _load_fonts()

    n_rows  = len(rows_data)
    n_cols  = 4
    total_w = row_label_w + n_cols * panel_w
    total_h = title_pad + n_rows * panel_h + (n_rows - 1) * row_gap + caption_h + footer

    out_img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw    = ImageDraw.Draw(out_img)

    title_txt = f"{asset.replace('_', ' ').capitalize()}  —  view {view_id}"
    bb = draw.textbbox((0, 0), title_txt, font=bold_font)
    tx = max(0, (total_w - (bb[2] - bb[0])) // 2)
    draw.text((tx, 6), title_txt, fill=(0, 0, 0), font=bold_font)

    for row_i, rd in enumerate(rows_data):
        y_top = title_pad + row_i * (panel_h + row_gap)

        for col_i, panel in enumerate(rd["panels"]):
            px = row_label_w + col_i * panel_w
            p  = panel.resize((panel_w, panel_h), Image.LANCZOS) \
                 if panel.size != (panel_w, panel_h) else panel
            out_img.paste(p, (px, y_top))

        # Row label: ablation name + key metrics, vertically centred
        m          = rd["metrics"]
        label_lines = [
            rd["abl_name"],
            f"CD {m['icp_cd_mm']:.1f}mm",
            f"F1 {m['icp_f1']:.2f}",
            f"fit {m['fitness']:.2f}",
        ]
        line_h   = 14
        total_lh = len(label_lines) * line_h
        ly_start = y_top + (panel_h - total_lh) // 2
        for li, line in enumerate(label_lines):
            lbb = draw.textbbox((0, 0), line, font=small_font)
            lw  = lbb[2] - lbb[0]
            lx  = max(2, (row_label_w - lw) // 2)
            draw.text((lx, ly_start + li * line_h), line, fill=(0, 0, 0), font=small_font)

        # no separator lines — matches sim eval style (panel boundaries only)

    # Column captions
    cap_y0     = title_pad + n_rows * panel_h + (n_rows - 1) * row_gap
    cap_labels = ["Part.", "Pred.", "KNN Err.", "GT+ICP Ov."]
    for col_i, label in enumerate(cap_labels):
        cbb = draw.textbbox((0, 0), label, font=reg_font)
        cw, ch = cbb[2] - cbb[0], cbb[3] - cbb[1]
        cx  = row_label_w + col_i * panel_w + (panel_w - cw) // 2
        cy  = cap_y0 + max(2, (caption_h - ch) // 2)
        draw.text((cx, cy), label, fill=(0, 0, 0), font=reg_font)

    # no divider lines — matches sim eval style (panel boundaries only)

    return out_img


# ─────────────────────────────────────────────
# Strip plots  (honest at n=5; shows all points + mean tick)
# ─────────────────────────────────────────────

def save_strip_plot(df, metric, ylabel, title, out_path, ablation_order=None):
    plt.rcParams.update(RCPARAMS)
    plot_df = df[np.isfinite(df[metric])].copy()

    if ablation_order:
        plot_df["ablation"] = pd.Categorical(
            plot_df["ablation"], categories=ablation_order, ordered=True)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    sns.stripplot(
        data=plot_df, x="asset", y=metric,
        hue="ablation", hue_order=ablation_order,
        dodge=True, jitter=0.12, size=7, alpha=0.8, ax=ax,
    )

    palette    = sns.color_palette(n_colors=len(ablation_order) if ablation_order
                                   else plot_df["ablation"].nunique())
    abl_list   = ablation_order or sorted(plot_df["ablation"].unique())
    asset_list = list(plot_df["asset"].cat.categories
                      if hasattr(plot_df["asset"], "cat")
                      else sorted(plot_df["asset"].unique()))
    n_abls     = len(abl_list)
    dodge_w    = 0.8 / n_abls

    for ai, abl in enumerate(abl_list):
        color  = palette[ai]
        offset = -0.4 + dodge_w * (ai + 0.5)
        for xi, asset in enumerate(asset_list):
            sub = plot_df[(plot_df["asset"] == asset) & (plot_df["ablation"] == abl)]
            if sub.empty:
                continue
            mean_val = sub[metric].mean()
            ax.plot([xi + offset - dodge_w * 0.35,
                     xi + offset + dodge_w * 0.35],
                    [mean_val, mean_val],
                    color=color, linewidth=2.0, solid_capstyle="round", zorder=5)

    for i in range(len(asset_list) - 1):
        ax.axvline(x=i + 0.5, color="lightgrey", linewidth=1, linestyle="--", zorder=0)

    ax.set_xlabel("Asset", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(title="Ablation", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────

def save_latex_table(df, out_path):
    """mean ± std per (asset, ablation) across the n views."""
    rows_tex = []
    for (asset, ablation), g in df.groupby(["asset", "ablation"], sort=False):
        def fmt(col):
            return f"{g[col].mean():.2f} $\\pm$ {g[col].std():.2f}"
        rows_tex.append(
            f"  {asset} & {ablation} & {fmt('icp_cd_mm')} & {fmt('icp_f1')} "
            f"& {fmt('fitness')} & {fmt('inlier_rmse_mm')} \\\\"
        )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Zero-Shot Real-World Evaluation (ICP Heuristic, mean\,$\pm$\,std)}",
        r"\label{tab:zeroshot_icp}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Asset & Ablation & CD\textsubscript{ICP} (mm) & F1\textsubscript{ICP} "
        r"& ICP Fitness & ICP RMSE (mm) \\",
        r"\midrule",
    ] + rows_tex + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def run_zero_shot(ablation_configs: list[dict],
                  txt_path: str,
                  data_dir: str,
                  mesh_specs: list[tuple[str, str]],
                  out_path: str,
                  gt_n_points: int     = 8192,
                  gt_seed: int         = 0,
                  panel_size: int      = 512,
                  grid_panel_size: int = 384,
                  point_size: float    = 2.0,
                  icp_max_dist: float  = 0.1,
                  icp_max_iter: int    = 100):
    """
    Args:
        mesh_specs   : [(taxonomy_id, mesh_path), ...]  one per asset.
                       GT is sampled once via Poisson disk (gt_n_points=8192)
                       and reused for every view of that asset.
        data_dir     : root containing <tid>/<mid>/models/<vid>.pcd
        txt_path     : lines of form <tid>-<mid>-<vid>.pcd
        icp_max_dist : ICP inlier threshold — must be in the same units as your
                       point clouds.  If clouds are in metres start around 0.05;
                       if in mm start around 50.
    """
    # ── GT ────────────────────────────────────────────────────────────────────
    print("Sampling GT from meshes...")
    gt_dict = build_gt_dict(mesh_specs, n_points=gt_n_points, seed=gt_seed)
    print()

    # ── Models ───────────────────────────────────────────────────────────────
    print("Loading models...")
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    predictors = []
    for abl in ablation_configs:
        print(f"  Loading '{abl['name']}'...")
        config = cfg_from_yaml_file(abl["cfg_path"])
        model  = builder.model_builder(config.model)
        builder.load_model(model, abl["ckpt_path"])
        model.to(device)
        model.eval()
        predictors.append((abl["name"], AdaPoinTrPredictor(model, normalize=False)))
    print(f"  {len(predictors)} model(s) loaded.\n")

    ablation_order = [a["name"] for a in ablation_configs]

    # ── Test list ─────────────────────────────────────────────────────────────
    with open(txt_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    file_list = []
    for line in lines:
        tid = line.split("-")[0].split("/")[-1]
        mid = line.split("-")[1]
        vid = int(line.split("-")[2].split(".")[0])
        file_list.append({"taxonomy_id": tid, "model_id": mid, "view_id": vid})

    print(f"  {len(file_list)} samples to process.\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_records = []
    view_rows   = defaultdict(list)   # (tid, vid) → list of row dicts

    for sample in file_list:
        tid = sample["taxonomy_id"]
        mid = sample["model_id"]
        vid = sample["view_id"]

        partial_path = os.path.join(data_dir, tid, mid, "models", f"{vid}.pcd")
        if not os.path.exists(partial_path):
            print(f"  [WARN] Partial not found: {partial_path}")
            continue

        if tid not in gt_dict:
            print(f"  [WARN] No mesh registered for '{tid}' — skipping")
            continue

        partial_pts = IO.get(partial_path).astype(np.float32)
        gt_pts      = gt_dict[tid]   # same cloud reused for all views of this asset

        for abl_name, predictor in predictors:
            print(f"  {tid}  view {vid}  [{abl_name}]")

            partial_norm, centroid = _norm_from_partial(partial_pts)
            complete_pts           = predictor.predict(partial_norm) * 2.0 + centroid

            # Camera anchored on combined partial+completion bbox so Pred. never clips
            anchor_pcd = o3d.geometry.PointCloud()
            anchor_pcd.points = o3d.utility.Vector3dVector(
                np.concatenate([partial_pts, complete_pts], axis=0).astype(np.float64))
            cam_params = _make_camera_params(anchor_pcd, fov_deg=60.0,
                                              width=panel_size, height=panel_size)

            icp_result  = run_icp(complete_pts, gt_pts,
                                   max_correspondence_dist=icp_max_dist,
                                   max_iter=icp_max_iter)
            icp_metrics = compute_icp_metrics(icp_result["registered_pts"], gt_pts)

            metrics = {
                "fitness":        icp_result["fitness"],
                "inlier_rmse_mm": icp_result["inlier_rmse_mm"],
                "icp_cd_mm":      icp_metrics["icp_cd_mm"],
                "icp_f1":         icp_metrics["icp_f1"],
            }

            panels, err_mean, err_max = _build_4_panels(
                partial_pts, complete_pts, gt_pts,
                icp_result["registered_pts"],
                cam_params, panel_size, panel_size, point_size,
            )

            # Individual 2×2 grid
            abl_tag   = "".join(abl_name.strip().lower().split())
            title_txt = (
                f"{tid.replace('_',' ').capitalize()}  view {vid}  [{abl_name}]  |  "
                f"CD {metrics['icp_cd_mm']:.1f}mm  F1 {metrics['icp_f1']:.2f}  "
                f"fit {metrics['fitness']:.2f}  RMSE {metrics['inlier_rmse_mm']:.2f}mm"
            )
            grid_2x2 = compose_2x2_grid(panels, title_txt,
                                         panel_w=panel_size, panel_h=panel_size,
                                         err_mean=err_mean, err_max=err_max)
            ind_path = os.path.join(out_path, "grids", "individual",
                                    abl_tag, tid, f"view_{vid:04d}.pdf")
            os.makedirs(os.path.dirname(ind_path), exist_ok=True)
            grid_2x2.save(ind_path)
            print(f"    Saved {ind_path}")

            view_rows[(tid, vid)].append({
                "abl_name": abl_name,
                "panels":   panels,
                "metrics":  metrics,
            })

            all_records.append({
                "asset":          tid,
                "ablation":       abl_name,
                "view_id":        vid,
                "fitness":        metrics["fitness"],
                "inlier_rmse_mm": metrics["inlier_rmse_mm"],
                "icp_cd_mm":      metrics["icp_cd_mm"],
                "icp_f1":         metrics["icp_f1"],
            })

    # ── Combined ablation-row grids ───────────────────────────────────────────
    for (tid, vid), rows in view_rows.items():
        combined  = compose_ablation_row_grid(
            rows, asset=tid, view_id=vid,
            panel_w=grid_panel_size, panel_h=grid_panel_size,
        )
        comb_path = os.path.join(out_path, "grids", "combined", tid, f"view_{vid:04d}.pdf")
        os.makedirs(os.path.dirname(comb_path), exist_ok=True)
        combined.save(comb_path)
        print(f"  Saved {comb_path}")

    # ── DataFrame + CSV ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df["asset"] = pd.Categorical(df["asset"],
                                  categories=sorted(df["asset"].unique()), ordered=True)
    csv_path = os.path.join(out_path, "combined_results.csv")
    os.makedirs(out_path, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")

    # ── Strip plots ───────────────────────────────────────────────────────────
    strip_dir = os.path.join(out_path, "plots", "strips")
    for col, ylabel, title in [
        ("icp_cd_mm",      "ICP CD (mm)",         "Chamfer Distance after ICP"),
        ("icp_f1",         "ICP F1",               "F1 Score after ICP"),
        ("fitness",        "ICP Fitness",          "ICP Registration Fitness"),
        ("inlier_rmse_mm", "ICP Inlier RMSE (mm)", "ICP Inlier RMSE"),
    ]:
        save_strip_plot(df, col, ylabel, title,
                        os.path.join(strip_dir, f"{col}.pdf"),
                        ablation_order=ablation_order)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    save_latex_table(df, os.path.join(out_path, "latex_table.tex"))

    print(f"\nAll done. Results in {out_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot real-world evaluation with ICP heuristic."
    )
    parser.add_argument(
        "--ablation", action="append", nargs=3,
        metavar=("NAME", "CFG", "CKPT"),
        help="Repeat for each ablation: NAME cfg_path ckpt_path",
    )
    parser.add_argument(
        "--mesh", action="append", nargs=2,
        metavar=("TAXONOMY_ID", "MESH_PATH"),
        help=(
            "Mesh for one asset: TAXONOMY_ID mesh_path.  Repeat for each asset.\n"
            "  --mesh glovebox    meshes/glovebox.stl\n"
            "  --mesh officechair meshes/officechair.stl\n"
            "  --mesh woodentable meshes/woodentable.stl\n"
            "  --mesh trashcan    meshes/trashcan.stl"
        ),
    )
    parser.add_argument("--txt_path",  required=True,
                        help="Test list txt — lines: taxonomy_id-model_id-view_id.pcd")
    parser.add_argument("--data_dir",  required=True,
                        help="Root of projected_partial_noise/ "
                             "(<data_dir>/<tid>/<mid>/models/<vid>.pcd)")
    parser.add_argument("--out_path",       default="./zeroshot_results")
    parser.add_argument("--gt_n_points",    type=int,   default=8192,
                        help="Poisson disk sample count for GT (default: 8192)")
    parser.add_argument("--gt_seed",        type=int,   default=0)
    parser.add_argument("--panel_size",     type=int,   default=512)
    parser.add_argument("--grid_panel_size",type=int,   default=384,
                        help="Panel size in the combined ablation-row grid")
    parser.add_argument("--point_size",     type=float, default=2.0)
    parser.add_argument("--icp_max_dist",   type=float, default=0.1,
                        help="ICP max correspondence distance (same units as point clouds)")
    parser.add_argument("--icp_max_iter",   type=int,   default=100)
    args = parser.parse_args()

    if not args.ablation:
        parser.error("Provide at least one --ablation NAME CFG CKPT")
    if not args.mesh:
        parser.error("Provide at least one --mesh TAXONOMY_ID MESH_PATH")

    ablation_configs = [
        {"name": a[0], "cfg_path": a[1], "ckpt_path": a[2]}
        for a in args.ablation
    ]
    mesh_specs = [(m[0], m[1]) for m in args.mesh]

    run_zero_shot(
        ablation_configs = ablation_configs,
        txt_path         = args.txt_path,
        data_dir         = args.data_dir,
        mesh_specs       = mesh_specs,
        out_path         = args.out_path,
        gt_n_points      = args.gt_n_points,
        gt_seed          = args.gt_seed,
        panel_size       = args.panel_size,
        grid_panel_size  = args.grid_panel_size,
        point_size       = args.point_size,
        icp_max_dist     = args.icp_max_dist,
        icp_max_iter     = args.icp_max_iter,
    )