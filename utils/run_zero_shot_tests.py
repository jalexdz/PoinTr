import argparse
import os
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
# Per (asset, view, ablation) produces a 2×2 panel grid:
#
#   Title: Asset -- view N  |  CD: xx mm  F1: xx  |  ICP fitness: xx  RMSE: xx mm
#   ┌──────────┬──────────┐
#   │  Part.   │  Pred.   │   ← top row
#   ├──────────┼──────────┤
#   │ KNN Err. │GT+ICP Ov.│   ← bottom row
#   └──────────┴──────────┘
#
# All views per asset are also combined into a single N-ablation-row grid:
#
#   ┌────────────┬──────────┬──────────┬──────────┬──────────┐
#   │  Baseline  │  Part.   │  Pred.   │ KNN Err. │GT+ICP Ov.│
#   ├────────────┼──────────┼──────────┼──────────┼──────────┤
#   │  Ablation1 │  ...     │  ...     │  ...     │  ...     │
#   └────────────┴──────────┴──────────┴──────────┴──────────┘
#
# Outputs:
#   <out_path>/plots/strips/         — combined strip plots (x=asset, hue=ablation)
#   <out_path>/grids/<asset>/        — per-view combined ablation grids
#   <out_path>/combined_results.csv  — all metrics
#   <out_path>/latex_table.tex       — ready-to-paste LaTeX table
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
# Normalization
# ─────────────────────────────────────────────

def _norm_from_partial(partial: np.ndarray):
    centroid     = np.mean(partial, axis=0)
    partial_norm = (partial - centroid) / 2.0
    return partial_norm.astype(np.float32), centroid


# ─────────────────────────────────────────────
# ICP
# ─────────────────────────────────────────────

def run_icp(source_pts: np.ndarray, target_pts: np.ndarray,
            max_correspondence_dist: float = 0.1,
            max_iter: int = 100) -> dict:
    """
    Register source → target using point-to-point ICP.
    Centroid pre-alignment is applied before ICP to aid convergence.

    Args:
        source_pts: (N, 3) predicted completion in world coords (mm scale)
        target_pts: (M, 3) sim GT in world coords (mm scale)
        max_correspondence_dist: ICP inlier threshold (same units as pts)
        max_iter: ICP iteration cap

    Returns dict with keys:
        registered_pts   — (N, 3) transformed source points
        fitness          — fraction of inlier correspondences [0, 1]
        inlier_rmse_mm   — RMSE of inlier pairs (mm)
        transformation   — (4, 4) homogeneous transform
    """
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source_pts.astype(np.float64))

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target_pts.astype(np.float64))

    # Centroid pre-alignment
    src_c = np.mean(source_pts, axis=0)
    tgt_c = np.mean(target_pts, axis=0)
    T_init = np.eye(4)
    T_init[:3, 3] = tgt_c - src_c

    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=max_correspondence_dist,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )

    src_pcd.transform(result.transformation)
    registered_pts = np.asarray(src_pcd.points).astype(np.float32)

    return {
        "registered_pts":  registered_pts,
        "fitness":         float(result.fitness),
        "inlier_rmse_mm":  float(result.inlier_rmse),
        "transformation":  result.transformation,
    }


# ─────────────────────────────────────────────
# Post-ICP CD + F1
# ─────────────────────────────────────────────

def compute_icp_metrics(registered_pts: np.ndarray, gt_pts: np.ndarray,
                         f1_threshold: float = 0.01) -> dict:
    """
    Compute CD and F1 between ICP-registered completion and sim GT.
    Uses the same Metrics.get() call as the sim eval so numbers are comparable.

    f1_threshold should match the threshold used in your sim evaluation.
    """
    reg_t  = torch.from_numpy(registered_pts).float().cuda().unsqueeze(0)
    gt_t   = torch.from_numpy(gt_pts).float().cuda().unsqueeze(0)
    m      = Metrics.get(reg_t, gt_t, require_emd=False)
    # m[0] = F1, m[1] = CD_L1 (normalized), convert to mm with ×2
    return {
        "icp_cd_mm": float(2 * m[1]),
        "icp_f1":    float(m[0]),
    }


# ─────────────────────────────────────────────
# Camera helpers
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


def _knn_error_colormap(pred_pcd, gt_pcd, cmap="viridis", vmax=None):
    """GT points coloured by distance to nearest predicted point."""
    pred_np = np.asarray(pred_pcd.points)
    gt_np   = np.asarray(gt_pcd.points)
    if pred_np.size == 0 or gt_np.size == 0:
        return np.zeros((gt_np.shape[0], 3)), np.zeros(gt_np.shape[0])
    tree  = o3d.geometry.KDTreeFlann(pred_pcd)
    dists = np.zeros(gt_np.shape[0], dtype=np.float32)
    for i, p in enumerate(gt_np):
        _, _, d2 = tree.search_knn_vector_3d(p, 1)
        dists[i] = np.sqrt(d2[0]) if len(d2) > 0 else 0.0
    vmax   = vmax or max(1e-6, float(np.percentile(dists, 95)))
    norm   = np.clip(dists / vmax, 0.0, 1.0)
    colors = plt.get_cmap(cmap)(norm)[:, :3]
    return colors.astype(np.float64), dists


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
# 2×2 panel builder for one (view, ablation)
# ─────────────────────────────────────────────

def _build_2x2_panels(partial_pts, complete_pts, gt_pts,
                       registered_pts, icp_result, icp_metrics,
                       cam_params, panel_w, panel_h, point_size):
    """
    Returns four PIL Images:
        [0] Part.        — partial cloud (grey)
        [1] Pred.        — raw completion (green)
        [2] KNN Err.     — GT coloured by dist to raw completion (viridis)
        [3] GT + ICP Ov. — GT (grey) + registered completion (green)
    """
    # ── PCDs ──────────────────────────────────────────────────────────────────
    part_pcd = o3d.geometry.PointCloud()
    part_pcd.points = o3d.utility.Vector3dVector(partial_pts.astype(np.float64))
    part_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.4, 0.4, 0.4], (partial_pts.shape[0], 1)))

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(complete_pts.astype(np.float64))
    pred_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 0.8, 0.0], (complete_pts.shape[0], 1)))

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pts.astype(np.float64))
    gt_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.6, 0.6, 0.6], (gt_pts.shape[0], 1)))

    # KNN error map — GT coloured by distance to raw completion
    err_colors, _ = _knn_error_colormap(pred_pcd, gt_pcd)
    knn_pcd = o3d.geometry.PointCloud()
    knn_pcd.points = o3d.utility.Vector3dVector(gt_pts.astype(np.float64))
    knn_pcd.colors = o3d.utility.Vector3dVector(err_colors)

    # GT + ICP overlay — GT grey, registered completion green
    reg_pcd = o3d.geometry.PointCloud()
    reg_pcd.points = o3d.utility.Vector3dVector(registered_pts.astype(np.float64))
    reg_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 0.8, 0.0], (registered_pts.shape[0], 1)))

    overlay_pcd = gt_pcd + reg_pcd   # merge GT and registered into one cloud

    # ── Render ────────────────────────────────────────────────────────────────
    imgs = []
    for pcd in [part_pcd, pred_pcd, knn_pcd, overlay_pcd]:
        arr = _render(pcd, cam_params,
                      width=panel_w, height=panel_h, point_size=point_size)
        imgs.append(Image.fromarray(arr))

    return imgs


def compose_2x2_grid(panels, title_txt, caption_labels,
                     panel_w, panel_h,
                     title_pad=44, caption_h=28, footer=10):
    """
    Compose four panels into a 2×2 grid with title and captions.
    caption_labels: list of 4 strings [top-left, top-right, bot-left, bot-right]
    """
    bold_font, reg_font, _ = _load_fonts()

    total_w = 2 * panel_w
    total_h = title_pad + 2 * panel_h + caption_h + footer

    img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title — may be two lines if long; just centre it
    bb = draw.textbbox((0, 0), title_txt, font=bold_font)
    tx = max(0, (total_w - (bb[2] - bb[0])) // 2)
    draw.text((tx, 5), title_txt, fill=(0, 0, 0), font=bold_font)

    y0 = title_pad
    positions = [(0, y0), (panel_w, y0), (0, y0 + panel_h), (panel_w, y0 + panel_h)]
    for panel, (px, py) in zip(panels, positions):
        img.paste(panel, (px, py))

    # Captions
    cap_y_rows = [y0 + panel_h, y0 + 2 * panel_h]
    for i, label in enumerate(caption_labels):
        col = i % 2
        row = i // 2
        bb  = draw.textbbox((0, 0), label, font=reg_font)
        cw  = bb[2] - bb[0]
        ch  = bb[3] - bb[1]
        cx  = col * panel_w + (panel_w - cw) // 2
        cy  = cap_y_rows[row] + max(2, (caption_h - ch) // 2)
        draw.text((cx, cy), label, fill=(0, 0, 0), font=reg_font)

    # Vertical + horizontal dividers
    draw.line([(panel_w, y0), (panel_w, y0 + 2 * panel_h)],
              fill=(200, 200, 200), width=1)
    draw.line([(0, y0 + panel_h), (total_w, y0 + panel_h)],
              fill=(200, 200, 200), width=1)

    return img


# ─────────────────────────────────────────────
# Combined N-ablation-row grid for one view
# ─────────────────────────────────────────────

def compose_ablation_row_grid(rows_data, asset, view_id,
                               panel_w=384, panel_h=384,
                               point_size=2.0,
                               row_label_w=110, caption_h=28,
                               row_gap=5, title_pad=36, footer=10):
    """
    rows_data: list of dicts, one per ablation, each with keys:
        abl_name, panels (list of 4 PIL Images), metrics dict

    Produces:
        ┌────────────┬──────────┬──────────┬──────────┬──────────┐
        │  Title (centred)                                        │
        ├────────────┼──────────┼──────────┼──────────┼──────────┤
        │  Baseline  │  Part.   │  Pred.   │ KNN Err. │GT+ICP Ov.│
        ├────────────┼──────────┼──────────┼──────────┼──────────┤
        │  Abl. 1    │  ...                                       │
        └────────────┴──────────┴──────────┴──────────┴──────────┘
        │            │  Part.   │  Pred.   │ KNN Err. │GT+ICP Ov.│  ← captions
    """
    bold_font, reg_font, small_font = _load_fonts()

    n_rows   = len(rows_data)
    n_cols   = 4   # Part. | Pred. | KNN Err. | GT+ICP Ov.
    total_w  = row_label_w + n_cols * panel_w
    total_h  = title_pad + n_rows * panel_h + (n_rows - 1) * row_gap + caption_h + footer

    out_img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw    = ImageDraw.Draw(out_img)

    # Title
    title_txt = f"{asset.replace('_', ' ').capitalize()}  —  view {view_id}"
    bb = draw.textbbox((0, 0), title_txt, font=bold_font)
    tx = max(0, (total_w - (bb[2] - bb[0])) // 2)
    draw.text((tx, 6), title_txt, fill=(0, 0, 0), font=bold_font)

    for row_i, rd in enumerate(rows_data):
        y_top = title_pad + row_i * (panel_h + row_gap)

        # Paste 4 panels
        for col_i, panel in enumerate(rd["panels"]):
            px = row_label_w + col_i * panel_w
            # Resize panel to target size if needed (panels may be 512×512)
            if panel.size != (panel_w, panel_h):
                panel = panel.resize((panel_w, panel_h), Image.LANCZOS)
            out_img.paste(panel, (px, y_top))

        # Row label — ablation name, vertically centred, plus mini metrics
        abl_name = rd["abl_name"]
        m        = rd["metrics"]
        label_lines = [
            abl_name,
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

        # Horizontal separator between rows
        if row_i < n_rows - 1:
            sep_y = y_top + panel_h + row_gap // 2
            draw.line([(row_label_w, sep_y), (total_w, sep_y)],
                      fill=(200, 200, 200), width=1)

    # Column captions
    cap_y0     = title_pad + n_rows * panel_h + (n_rows - 1) * row_gap
    cap_labels = ["Part.", "Pred.", "KNN Err.", "GT+ICP Ov."]
    for col_i, label in enumerate(cap_labels):
        cbb = draw.textbbox((0, 0), label, font=reg_font)
        cw  = cbb[2] - cbb[0]
        ch  = cbb[3] - cbb[1]
        cx  = row_label_w + col_i * panel_w + (panel_w - cw) // 2
        cy  = cap_y0 + max(2, (caption_h - ch) // 2)
        draw.text((cx, cy), label, fill=(0, 0, 0), font=reg_font)

    # Vertical column dividers
    for col_i in range(1, n_cols):
        vx = row_label_w + col_i * panel_w
        draw.line([(vx, title_pad), (vx, cap_y0)], fill=(200, 200, 200), width=1)

    return out_img


# ─────────────────────────────────────────────
# Strip plots
# ─────────────────────────────────────────────

def save_strip_plot(df, metric, ylabel, title, out_path, ablation_order=None):
    """
    Strip plot: x=asset, hue=ablation, individual points shown.
    Mean per (asset, ablation) shown as a horizontal tick.
    Appropriate for n=5 — honest about small sample size.
    """
    plt.rcParams.update(RCPARAMS)
    plot_df = df[np.isfinite(df[metric])].copy()

    if ablation_order:
        plot_df["ablation"] = pd.Categorical(
            plot_df["ablation"], categories=ablation_order, ordered=True)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    sns.stripplot(
        data=plot_df, x="asset", y=metric,
        hue="ablation", hue_order=ablation_order,
        dodge=True, jitter=0.12, size=7, alpha=0.8,
        ax=ax,
    )

    # Mean tick per (asset, ablation) group
    palette = sns.color_palette(n_colors=len(ablation_order) if ablation_order else
                                plot_df["ablation"].nunique())
    abl_list   = ablation_order or sorted(plot_df["ablation"].unique())
    asset_list = list(plot_df["asset"].cat.categories
                      if hasattr(plot_df["asset"], "cat")
                      else sorted(plot_df["asset"].unique()))
    n_abls     = len(abl_list)
    dodge_w    = 0.8 / n_abls

    for ai, abl in enumerate(abl_list):
        color = palette[ai]
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

    n_assets = len(asset_list)
    for i in range(n_assets - 1):
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
    """
    One row per (asset, ablation).
    Columns: CD (ICP) mm, F1 (ICP), ICP fitness, ICP RMSE mm
    Format: mean ± std across the n_views views.
    """
    rows_tex = []
    for (asset, ablation), g in df.groupby(["asset", "ablation"], sort=False):
        def fmt(col):
            m = g[col].mean()
            s = g[col].std()
            return f"{m:.2f} $\\pm$ {s:.2f}"

        rows_tex.append(
            f"  {asset} & {ablation} & {fmt('icp_cd_mm')} & {fmt('icp_f1')} "
            f"& {fmt('fitness')} & {fmt('inlier_rmse_mm')} \\\\"
        )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Zero-Shot Real-World Evaluation (ICP Heuristic, mean\,$\pm$\,std over 5 views)}",
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
# Core inference + ICP loop
# ─────────────────────────────────────────────

def run_zero_shot(ablation_configs: list[dict],
                  txt_path: str,
                  data_dir: str,
                  gt_dir: str,
                  out_path: str,
                  panel_size: int   = 512,
                  grid_panel_size: int = 384,
                  point_size: float = 2.0,
                  icp_max_dist: float = 0.1,
                  icp_max_iter: int   = 100):
    """
    Main loop.

    Directory conventions:
        Partial PCDs : <data_dir>/<taxonomy_id>/<model_id>/models/<view_id>.pcd
        Sim GT PCDs  : <gt_dir>/<taxonomy_id>-<model_id>-<view_id>.pcd
                       (same naming as NRG_pc used in sim eval)

    txt_path line format: <taxonomy_id>-<model_id>-<view_id>.pcd
    """
    # ── Load models ───────────────────────────────────────────────────────────
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

    # ── Parse test list ───────────────────────────────────────────────────────
    with open(txt_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    file_list = []
    for line in lines:
        tid = line.split("-")[0].split("/")[-1]
        mid = line.split("-")[1]
        vid = int(line.split("-")[2].split(".")[0])
        file_list.append({"taxonomy_id": tid, "model_id": mid, "view_id": vid})

    print(f"  {len(file_list)} samples.\n")

    # ── Per-view inference ────────────────────────────────────────────────────
    all_records = []   # flat list of per-(view, ablation) metric dicts

    # Group by view so we can build the combined ablation-row grid per view
    from collections import defaultdict
    view_rows = defaultdict(list)   # key=(taxonomy_id, view_id) → list of row dicts

    for sample in file_list:
        tid = sample["taxonomy_id"]
        mid = sample["model_id"]
        vid = sample["view_id"]

        partial_path = os.path.join(data_dir, tid, mid, "models", f"{vid}.pcd")
        gt_path      = os.path.join(gt_dir, f"{tid}-{mid}-{vid}.pcd")

        if not os.path.exists(partial_path):
            print(f"  [WARN] Partial not found: {partial_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"  [WARN] GT not found: {gt_path}")
            continue

        partial_pts = IO.get(partial_path).astype(np.float32)
        gt_pts      = IO.get(gt_path).astype(np.float32)

        # Camera anchored on partial cloud, shared across all ablation rows
        anchor_pcd = o3d.geometry.PointCloud()
        anchor_pcd.points = o3d.utility.Vector3dVector(partial_pts.astype(np.float64))
        cam_params = _make_camera_params(anchor_pcd, fov_deg=60.0,
                                          width=panel_size, height=panel_size)

        for abl_name, predictor in predictors:
            print(f"  {tid} view {vid}  [{abl_name}]")

            partial_norm, centroid = _norm_from_partial(partial_pts)
            complete_pts           = predictor.predict(partial_norm) * 2.0 + centroid

            # ICP: register completion → sim GT
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

            # Build 4 panels
            panels = _build_2x2_panels(
                partial_pts, complete_pts, gt_pts,
                icp_result["registered_pts"], icp_result, icp_metrics,
                cam_params, panel_size, panel_size, point_size,
            )

            # ── Individual 2×2 grid ───────────────────────────────────────────
            abl_tag   = "".join(abl_name.strip().lower().split())
            title_txt = (f"{tid.replace('_',' ').capitalize()}  view {vid}  [{abl_name}]  |  "
                         f"CD {metrics['icp_cd_mm']:.1f}mm  F1 {metrics['icp_f1']:.2f}  "
                         f"fit {metrics['fitness']:.2f}  RMSE {metrics['inlier_rmse_mm']:.2f}mm")

            grid_2x2 = compose_2x2_grid(
                panels, title_txt,
                caption_labels=["Part.", "Pred.", "KNN Err.", "GT+ICP Ov."],
                panel_w=panel_size, panel_h=panel_size,
            )
            ind_path = os.path.join(out_path, "grids", "individual",
                                    abl_tag, tid, f"view_{vid:04d}.pdf")
            os.makedirs(os.path.dirname(ind_path), exist_ok=True)
            grid_2x2.save(ind_path)
            print(f"    Saved {ind_path}")

            # Accumulate for combined ablation-row grid
            view_rows[(tid, vid)].append({
                "abl_name": abl_name,
                "panels":   panels,
                "metrics":  metrics,
            })

            # Accumulate for strip plots + CSV
            all_records.append({
                "asset":          tid,
                "ablation":       abl_name,
                "view_id":        vid,
                "fitness":        metrics["fitness"],
                "inlier_rmse_mm": metrics["inlier_rmse_mm"],
                "icp_cd_mm":      metrics["icp_cd_mm"],
                "icp_f1":         metrics["icp_f1"],
            })

    # ── Combined ablation-row grids (one per view) ────────────────────────────
    for (tid, vid), rows in view_rows.items():
        combined = compose_ablation_row_grid(
            rows, asset=tid, view_id=vid,
            panel_w=grid_panel_size, panel_h=grid_panel_size,
            point_size=point_size,
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
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")

    # ── Strip plots ───────────────────────────────────────────────────────────
    strip_dir = os.path.join(out_path, "plots", "strips")
    metrics_cfg = [
        ("icp_cd_mm",      "ICP CD (mm)",         "Chamfer Distance after ICP Registration"),
        ("icp_f1",         "ICP F1",               "F1 Score after ICP Registration"),
        ("fitness",        "ICP Fitness",          "ICP Registration Fitness"),
        ("inlier_rmse_mm", "ICP Inlier RMSE (mm)", "ICP Inlier RMSE"),
    ]
    for col, ylabel, title in metrics_cfg:
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
    parser.add_argument("--txt_path",  required=True,
                        help="Test list txt (taxonomy-model-view.pcd lines)")
    parser.add_argument("--data_dir",  required=True,
                        help="Root of projected_partial_noise/")
    parser.add_argument("--gt_dir",    required=True,
                        help="Directory of sim GT PCDs (taxonomy-model-view.pcd)")
    parser.add_argument("--out_path",  default="./zeroshot_results")
    parser.add_argument("--panel_size",      type=int,   default=512)
    parser.add_argument("--grid_panel_size", type=int,   default=384,
                        help="Panel size in the combined ablation-row grid")
    parser.add_argument("--point_size",      type=float, default=2.0)
    parser.add_argument("--icp_max_dist",    type=float, default=0.1,
                        help="ICP max correspondence distance (same units as point clouds)")
    parser.add_argument("--icp_max_iter",    type=int,   default=100)
    args = parser.parse_args()

    if not args.ablation:
        parser.error("Provide at least one --ablation NAME CFG CKPT")

    ablation_configs = [
        {"name": a[0], "cfg_path": a[1], "ckpt_path": a[2]}
        for a in args.ablation
    ]

    run_zero_shot(
        ablation_configs  = ablation_configs,
        txt_path          = args.txt_path,
        data_dir          = args.data_dir,
        gt_dir            = args.gt_dir,
        out_path          = args.out_path,
        panel_size        = args.panel_size,
        grid_panel_size   = args.grid_panel_size,
        point_size        = args.point_size,
        icp_max_dist      = args.icp_max_dist,
        icp_max_iter      = args.icp_max_iter,
    )