import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np
import torch
import open3d as o3d
from datasets.io import IO
from PIL import Image, ImageDraw, ImageFont

from tools import builder
from utils.config import cfg_from_yaml_file
from tools.predictor import AdaPoinTrPredictor
from utils.metrics import Metrics


# ─────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────

def _norm_from_partial(gt, partial):
    centroid = np.mean(partial, axis=0)
    p0 = partial - centroid
    partial0 = p0 / 2.0
    gt0 = (gt - centroid) / 2.0
    return gt0.astype(np.float32), partial0.astype(np.float32), centroid


# ─────────────────────────────────────────────
# DataFrame helpers
# ─────────────────────────────────────────────

def per_sample_to_df(per_sample, ablation_name):
    rows = []
    for asset, entries in per_sample.items():
        for e in entries:
            rows.append({
                "asset":                      asset,
                "ablation":                   ablation_name,
                "view_id":                    e["idx"],
                "cd":                         float(e["cd"]),
                "emd":                        float(e["emd"]),
                "f1":                         float(e["f1"]),
                "partial_to_gt_mean":         float(e["partial_to_gt_mean"]),
                "partial_to_gt_std":          float(e["partial_to_gt_std"]),
                "partial_to_gt_median":       float(e["partial_to_gt_median"]),
                "completion_to_gt_mean":      float(e["completion_to_gt_mean"]),
                "completion_to_gt_std":       float(e["completion_to_gt_std"]),
                "completion_to_gt_median":    float(e["completion_to_gt_median"]),
            })
    df = pd.DataFrame(rows)
    df["asset"] = pd.Categorical(
        df["asset"],
        categories=sorted(df["asset"].unique()),
        ordered=True,
    )
    return df


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────

RCPARAMS = {
    "font.size":        25,
    "axes.labelsize":   22,
    "axes.titlesize":   25,
    "xtick.labelsize":  20,
    "ytick.labelsize":  20,
    "legend.fontsize":  20,
    "axes.ymargin":     0.08,   # tight top/bottom padding (default ~0.10)
}

# Shared kwargs for all sns.boxplot calls — thicker lines, visible at 600 dpi
_BOX_PROPS = dict(
    boxprops=dict(linewidth=1.25),
    whiskerprops=dict(linewidth=1.25),
    capprops=dict(linewidth=1.25),
    medianprops=dict(linewidth=1.25, color="black"),
)


def save_single_ablation_boxplot(df, metric, out_path, title, ylabel):
    """Single-ablation boxplot (no hue). One observation per viewpoint."""
    assert metric in df.columns, f"Invalid metric: {metric}"
    plt.rcParams.update(RCPARAMS)

    plot_df = df[np.isfinite(df[metric])].copy()
    fig, ax = plt.subplots(figsize=(4.5, 5.5), constrained_layout=True)

    sns.boxplot(
        data=plot_df, x="asset", y=metric,
        showfliers=True, whis=1.5, ax=ax,
        flierprops=dict(marker='d', markerfacecolor='black', markersize=3),
        **_BOX_PROPS,
    )

    # grid
    ax.yaxis.grid(True, linestyle='--', linewidth=0.8, color='lightgrey', zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlabel('Asset Class', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title or f'{metric.upper()} by Asset', fontsize=20)
    plt.xticks(rotation=20, ha="right")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def save_ablation_boxplot(df, metric, out_path, title, ylabel, ablation_order=None):
    """
    Multi-ablation grouped boxplot (x=asset, hue=ablation).
    One observation per viewpoint — same granularity as CD/F1.
    """
    assert metric in df.columns, f"Invalid metric: {metric}"
    plt.rcParams.update(RCPARAMS)

    plot_df = df[np.isfinite(df[metric])].copy()

    if ablation_order is not None:
        plot_df["ablation"] = pd.Categorical(
            plot_df["ablation"], categories=ablation_order, ordered=True
        )

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)

    sns.boxplot(
        data=plot_df, x="asset", y=metric,
        hue="ablation", hue_order=ablation_order,
        showfliers=True, whis=1.5, gap=0.15, width=0.7,
        ax=ax,
        flierprops=dict(marker='d', markerfacecolor='black', markersize=3),
        **_BOX_PROPS,
    )

    # horizontal grid behind boxes
    ax.yaxis.grid(True, linestyle='--', linewidth=0.8, color='lightgrey', zorder=0)
    ax.set_axisbelow(True)

    # vertical separators between asset groups
    n_assets = plot_df["asset"].nunique()
    for i in range(n_assets - 1):
        ax.axvline(x=i + 0.5, color='lightgrey', linewidth=1.0, linestyle='--', zorder=0)

    ax.set_xlabel('Asset Class', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.legend(title='Ablation', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=20, ha="right")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def save_single_denoising_boxplot(partial_mean_by_asset, completion_mean_by_asset, out_path):
    """
    Per-ablation denoising boxplot.
    Each observation = per-viewpoint mean point-to-GT distance (mm).
    x=Asset, hue=Source (Partial vs Completion).
    """
    records = []
    for asset in sorted(partial_mean_by_asset.keys()):
        for d in partial_mean_by_asset[asset]:
            records.append({'Asset': asset, 'Source': 'Partial',
                            'Mean Error (mm)': float(d) * 2})
        for d in completion_mean_by_asset[asset]:
            records.append({'Asset': asset, 'Source': 'Completion',
                            'Mean Error (mm)': float(d) * 2})

    plot_df = pd.DataFrame(records)
    plot_df = plot_df[np.isfinite(plot_df["Mean Error (mm)"])]

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    sns.boxplot(
        data=plot_df, x="Asset", y="Mean Error (mm)", hue="Source",
        ax=ax, whis=1.5, gap=0.3,
        flierprops=dict(marker='d', markerfacecolor='black', markersize=3),
        **_BOX_PROPS,
    )

    ax.yaxis.grid(True, linestyle='--', linewidth=0.8, color='lightgrey', zorder=0)
    ax.set_axisbelow(True)

    n_assets = plot_df["Asset"].nunique()
    for i in range(n_assets - 1):
        ax.axvline(x=i + 0.5, color='lightgrey', linewidth=1.0, linestyle='--', zorder=0)

    ax.set_xlabel('Asset Class', fontsize=18)
    ax.set_ylabel('Mean Point-to-GT Distance (mm)', fontsize=18)
    ax.set_title('Partial vs. Completion Denoising Error (per-viewpoint mean)', fontsize=20)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def _build_denoising_records(all_denoising):
    """Shared helper: build long-form DataFrame from all_denoising list."""
    records = []
    for entry in all_denoising:
        abl = entry['name']
        for asset in sorted(entry['partial_means'].keys()):
            for d in entry['partial_means'][asset]:
                records.append({'Asset': asset, 'Ablation': abl,
                                 'Source': 'Partial',    'Mean Error (mm)': float(d) * 2})
            for d in entry['completion_means'][asset]:
                records.append({'Asset': asset, 'Ablation': abl,
                                 'Source': 'Completion', 'Mean Error (mm)': float(d) * 2})
    return pd.DataFrame(records)


def save_combined_denoising_boxplot(all_denoising, out_path, ablation_order=None):
    """
    Layout A — 2 panels: Partial | Completion, hue=Ablation.
    Saved as: combined/denoising_by_source.pdf
    """
    plt.rcParams.update(RCPARAMS)
    plot_df = _build_denoising_records(all_denoising)
    plot_df = plot_df[np.isfinite(plot_df["Mean Error (mm)"])]

    if ablation_order:
        plot_df["Ablation"] = pd.Categorical(
            plot_df["Ablation"], categories=ablation_order, ordered=True
        )
    asset_order = sorted(plot_df["Asset"].unique())
    n_assets    = len(asset_order)
    plot_df["Asset"] = pd.Categorical(
        plot_df["Asset"], categories=asset_order, ordered=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 5), constrained_layout=True)
    for ax, source in zip(axes, ['Partial', 'Completion']):
        sub = plot_df[plot_df['Source'] == source]
        sns.boxplot(
            data=sub, x="Asset", y="Mean Error (mm)",
            hue="Ablation", hue_order=ablation_order,
            ax=ax, whis=1.5, gap=0.15, width=0.7,
            flierprops=dict(marker='d', markerfacecolor='black', markersize=3),
            **_BOX_PROPS,
        )
        ax.yaxis.grid(True, linestyle='--', linewidth=0.8, color='lightgrey', zorder=0)
        ax.set_axisbelow(True)
        for i in range(n_assets - 1):
            ax.axvline(x=i + 0.5, color='lightgrey', linewidth=1.0, linestyle='--', zorder=0)
        ax.set_title(f'{source} Point-to-GT Error', fontsize=20)
        ax.set_xlabel('Asset Class', fontsize=18)
        ax.set_ylabel('Mean Error (mm)', fontsize=18)
        ax.legend(title='Ablation', bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=20)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def save_combined_denoising_by_ablation_boxplot(all_denoising, out_path, ablation_order=None):
    """
    Layout B — N panels (one per ablation), each showing x=Asset, hue=Source.
    Saved as: combined/denoising_by_ablation.pdf
    """
    plt.rcParams.update(RCPARAMS)
    plot_df = _build_denoising_records(all_denoising)
    plot_df = plot_df[np.isfinite(plot_df["Mean Error (mm)"])]

    abl_list    = ablation_order if ablation_order else sorted(plot_df["Ablation"].unique())
    asset_order = sorted(plot_df["Asset"].unique())
    n_assets    = len(asset_order)
    n_abls      = len(abl_list)

    plot_df["Asset"] = pd.Categorical(
        plot_df["Asset"], categories=asset_order, ordered=True
    )

    ymin = 0
    ymax = plot_df["Mean Error (mm)"].quantile(0.99) * 1.1

    fig, axes = plt.subplots(1, n_abls, figsize=(7 * n_abls, 5),
                              constrained_layout=True, sharey=True)
    if n_abls == 1:
        axes = [axes]

    source_order = ['Partial', 'Completion']

    for ax, abl in zip(axes, abl_list):
        sub = plot_df[plot_df['Ablation'] == abl]
        sns.boxplot(
            data=sub, x="Asset", y="Mean Error (mm)",
            hue="Source", hue_order=source_order,
            ax=ax, whis=1.5, gap=0.2, width=0.65,
            flierprops=dict(marker='d', markerfacecolor='black', markersize=3),
            **_BOX_PROPS,
        )
        ax.yaxis.grid(True, linestyle='--', linewidth=0.8, color='lightgrey', zorder=0)
        ax.set_axisbelow(True)
        for i in range(n_assets - 1):
            ax.axvline(x=i + 0.5, color='lightgrey', linewidth=1.0, linestyle='--', zorder=0)

        ax.set_title(abl, fontsize=20, fontweight='bold')
        ax.set_xlabel('Asset Class', fontsize=18)
        ax.set_ylabel('Mean Point-to-GT Error (mm)' if ax == axes[0] else '', fontsize=18)
        ax.set_ylim(ymin, ymax)
        ax.legend(title='Source', bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=20)

    fig.suptitle('Partial vs. Completion Point-to-GT Error by Ablation', fontsize=18)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────

def _compute_error_colormap(pred, gt, cmap='turbo', vmax=None):
    pred_np = np.asarray(pred.points)
    gt_np   = np.asarray(gt.points)
    if pred_np.size == 0 or gt_np.size == 0:
        return np.zeros((pred_np.shape[0], 3), dtype=np.float32), np.zeros((pred_np.shape[0],), dtype=np.float32)
    tree  = o3d.geometry.KDTreeFlann(pred)
    dists = np.zeros((gt_np.shape[0],), dtype=np.float32)
    for i, p in enumerate(gt_np):
        _, idx, dist2 = tree.search_knn_vector_3d(p, 1)
        dists[i] = 1000 * np.sqrt(dist2[0]) if len(dist2) > 0 else 0.0
    if vmax is None:
        vmax = 100  # mm
    norm   = np.clip(dists / float(vmax), 0.0, 1.0)
    colors = plt.get_cmap(cmap)(norm)[:, :3]
    return colors.astype(np.float64), dists


def _make_camera_params_from_gt(gt_pcd, cam_offset_factor=(0.8, -1.0, 1.6),
                                 fov_deg=60.0, width=512, height=512,
                                 point_size=3.5, visible=False):
    bbox   = gt_pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = float(np.linalg.norm(extent) * 0.5)
    radius = radius if radius > 0 else 1.0

    z  = 2 * radius
    xy = np.array([1.0, 0.0], dtype=np.float64) * radius
    th = np.deg2rad(45.0)
    Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float64)
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
    fx = fy  = 0.5 * width / np.tan(0.5 * fov_rad)
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, width * 0.5, height * 0.5)
    params.intrinsic        = intr
    params.intrinsic.width  = int(width)
    params.intrinsic.height = int(height)
    vis.destroy_window()

    return {"params": params, "center": center, "radius": radius,
            "cam_pos": cam_pos, "cam_up": cam_up}


def _render_with_camera_params(pcd, params, width=512, height=512,
                                point_size=3.5, visible=False):
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


def render_triplet_from_pcds(partial_pcd_path, gt_pcd_path, out_path,
                              predictor, asset, cd, f1, sel_type,
                              include_error=True, panel_size=(512, 512),
                              point_size=3.0, title_font_path=None):
    assert os.path.exists(partial_pcd_path), f"Partial PCD not found: {partial_pcd_path}"
    assert os.path.exists(gt_pcd_path),      f"GT PCD not found: {gt_pcd_path}"

    partial_pcd = o3d.io.read_point_cloud(partial_pcd_path)
    gt_pcd      = o3d.io.read_point_cloud(gt_pcd_path)
    gt_pts      = np.asarray(gt_pcd.points)

    input_raw = IO.get(partial_pcd_path).astype(np.float32)
    gt_raw    = IO.get(gt_pcd_path).astype(np.float32)
    gt_norm, input_norm, c = _norm_from_partial(gt_raw, input_raw)
    complete = predictor.predict(input_norm)
    complete = complete * 2.0 + c

    complete_pcd = o3d.geometry.PointCloud()
    complete_pcd.points = o3d.utility.Vector3dVector(complete)
    complete_pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.0, 1.0, 0.0], (complete.shape[0], 1))
    )

    w, h   = panel_size
    anchor = _make_camera_params_from_gt(gt_pcd, fov_deg=60.0, width=w, height=h,
                                          point_size=point_size, visible=False)
    params = anchor['params']

    img_partial  = _render_with_camera_params(partial_pcd,  params, width=w, height=h, point_size=point_size)
    img_complete = _render_with_camera_params(complete_pcd, params, width=w, height=h, point_size=point_size)
    img_gt       = _render_with_camera_params(gt_pcd,       params, width=w, height=h, point_size=point_size)
    panels = [Image.fromarray(img_partial), Image.fromarray(img_complete), Image.fromarray(img_gt)]

    pred_error_stats = None
    if include_error:
        pred_pts = np.asarray(complete_pcd.points).copy()
        gt_pts_err = np.asarray(gt_pcd.points).copy()

        pred_pts -= pred_pts.mean(axis=0)
        gt_pts_err -= gt_pts_err.mean(axis=0)

        complete_pcd_err = o3d.geometry.PointCloud()
        complete_pcd_err.points = o3d.utility.Vector3dVector(pred_pts)

        gt_pcd_err = o3d.geometry.PointCloud()
        gt_pcd_err.points = o3d.utility.Vector3dVector(gt_pts_err)

        err_colors, dists = _compute_error_colormap(complete_pcd_err, gt_pcd_err, cmap='turbo')

        pred_err_pcd = o3d.geometry.PointCloud()
        pred_err_pcd.points = o3d.utility.Vector3dVector(gt_pts.astype(np.float64))
        pred_err_pcd.colors = o3d.utility.Vector3dVector(err_colors)

        img_err = _render_with_camera_params(pred_err_pcd, params, width=w, height=h, point_size=3.5)
        panels.append(Image.fromarray(img_err))

        pred_error_stats = {
            "mean": float(dists.mean()),
            "max":  float(np.max(dists)),
            "min":  float(np.min(dists)),
        }

    try:
        title_font   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        caption_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    except Exception:
        title_font = caption_font = ImageFont.load_default()

    pad_between    = 0
    grid_w         = 2 * w + pad_between
    grid_h         = 2 * h + pad_between
    caption_height = 30
    footer_padding = 30

    sel_label = {"lowest": "Best", "median": "Median", "highest": "Worst"}.get(
        sel_type, sel_type.capitalize() if "outlier" in sel_type else "Unknown"
    )
    cd_txt    = f"CD: {cd:.2f} mm" if cd is not None else "CD: N/A"
    f1_txt    = f"F1: {f1:.2f}"    if f1 is not None else "F1: N/A"
    title_txt = f"{asset.capitalize()} {sel_label} ({cd_txt}, {f1_txt})"

    tmp_draw       = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bbox_title     = tmp_draw.textbbox((0, 0), title_txt, font=title_font)
    title_text_h   = bbox_title[3] - bbox_title[1]
    title_pad_top  = 4
    title_pad_bot  = 6
    title_height   = title_text_h + title_pad_top + title_pad_bot
    final_w        = grid_w
    final_h        = title_height + grid_h + caption_height + footer_padding

    out_image = Image.new("RGB", (final_w, final_h), (255, 255, 255))
    draw      = ImageDraw.Draw(out_image)
    y_panels  = title_height

    out_image.paste(panels[0], (0, y_panels))
    out_image.paste(panels[1], (w, y_panels))
    out_image.paste(panels[3] if include_error else panels[2], (0, y_panels + h))
    out_image.paste(panels[2],                                  (w, y_panels + h))

    bbox    = draw.textbbox((0, 0), title_txt, font=title_font)
    title_x = max(0, (final_w - (bbox[2] - bbox[0])) // 2)
    draw.text((title_x, title_pad_top), title_txt, fill=(0, 0, 0), font=title_font)

    caption_labels = ["Part.", "Pred."]
    if include_error:
        caption_labels.append(
            f"Mean Err.:{2*pred_error_stats['mean']:.2f} mm\nMax Err.:{2*pred_error_stats['max']:.2f} mm"
        )
    caption_labels.append("GT")

    for i, label in enumerate(caption_labels):
        row_i, col_i = i // 2, i % 2
        cell_x       = col_i * (w + pad_between)
        cell_y_top   = y_panels + row_i * (h + pad_between)
        bbox_c       = draw.textbbox((0, 0), label, font=caption_font)
        cap_w        = bbox_c[2] - bbox_c[0]
        cap_h        = bbox_c[3] - bbox_c[1]
        cap_x        = cell_x + (w - cap_w) // 2
        cap_y        = cell_y_top + h + max(2, (caption_height - cap_h) // 2)
        draw.text((cap_x, cap_y), label, fill=(0, 0, 0), font=caption_font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_image.save(out_path)
    print(f"  Render saved {out_path}")
    return out_path

def compute_outlier_summary(df, metric="cd"):
    """
    Compute per-asset outlier counts and percentages using the same 1.5*IQR rule
    already used in compute_samples().
    """
    assert metric in df.columns, f"Invalid metric: {metric}"

    rows = []
    for (ablation, asset), g in df.groupby(["ablation", "asset"]):
        vals = g[metric].dropna().values
        if len(vals) == 0:
            continue

        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        is_outlier = (g[metric] < lower) | (g[metric] > upper)
        n_total = int(len(g))
        n_outliers = int(is_outlier.sum())
        pct_outliers = 100.0 * n_outliers / max(n_total, 1)

        rows.append({
            "ablation": ablation,
            "asset": asset,
            "metric": metric,
            "n_total": n_total,
            "n_outliers": n_outliers,
            "pct_outliers": pct_outliers,
        })

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df["asset"] = pd.Categorical(
            out_df["asset"],
            categories=sorted(out_df["asset"].unique()),
            ordered=True,
        )
    return out_df


def save_outlier_percentage_barplot(outlier_df, out_path, metric="cd", ablation_order=None):
    """
    Grouped bar chart: x=asset, y=% outliers, hue=ablation.
    """
    if outlier_df.empty:
        print("[WARN] No outlier data to plot.")
        return

    plot_df = outlier_df.copy()

    if ablation_order is not None:
        plot_df["ablation"] = pd.Categorical(
            plot_df["ablation"], categories=ablation_order, ordered=True
        )

    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    sns.barplot(
        data=plot_df,
        x="asset",
        y="pct_outliers",
        hue="ablation",
        hue_order=ablation_order,
        ax=ax,
        errorbar=None,
    )

    ax.yaxis.grid(True, linestyle='--', linewidth=0.8, color='lightgrey', zorder=0)
    ax.set_axisbelow(True)

    n_assets = plot_df["asset"].nunique()
    for i in range(n_assets - 1):
        ax.axvline(x=i + 0.5, color='lightgrey', linewidth=1.0, linestyle='--', zorder=0)

    ax.set_xlabel("Asset Class", fontsize=18)
    ax.set_ylabel("Outliers (%)", fontsize=18)
    ax.set_title(f"Outlier Percentage by Asset and Ablation ({metric.upper()})", fontsize=20)
    ax.set_ylim(0, max(5, plot_df["pct_outliers"].max() * 1.15))
    ax.legend(title="Ablation", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")

    # optional value labels
    for container in ax.containers:
        labels = []
        for bar in container:
            h = bar.get_height()
            labels.append(f"{h:.1f}" if np.isfinite(h) else "")
        ax.bar_label(container, labels=labels, padding=2, fontsize=14)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

# ─────────────────────────────────────────────
# Sample selection + graphic export
# ─────────────────────────────────────────────

def compute_samples(df, ablation_name, predictor,
                    out_csv, job_out_dir, metric_col="cd"):
    assert metric_col in df.columns

    picks = []
    for asset, g in df.groupby("asset"):
        vals = g[metric_col].dropna().values
        if len(vals) == 0:
            continue
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
        iqr         = q3 - q1
        non_outlier = g[(g[metric_col] >= q1 - 1.5*iqr) & (g[metric_col] <= q3 + 1.5*iqr)].copy()
        outlier_df  = g[(g[metric_col] <  q1 - 1.5*iqr) | (g[metric_col] >  q3 + 1.5*iqr)].copy()
        if len(non_outlier) == 0:
            non_outlier = g.copy()

        def closest(value):
            idx = (g[metric_col] - value).abs().idxmin()
            return idx, g.loc[idx]

        for kind, value in [("lowest",  non_outlier[metric_col].min()),
                             ("median",  q2),
                             ("highest", non_outlier[metric_col].max())]:
            try:
                idx, row = closest(value)
                picks.append((asset, metric_col, kind, idx, row))
            except Exception:
                print(f"[WARN] Couldn't pick {kind} for {asset}")

        for oi, orow in outlier_df.iterrows():
            picks.append((asset, metric_col, "outlier", oi, orow))

    pick_rows = []
    for asset, metric_used, sel_type, idx, row in picks:
        pick_row = {
            "asset": asset, "ablation": ablation_name,
            "selection_type": sel_type, "selection_metric": metric_used,
            "df_index": int(idx), "view_id": int(row["view_id"]),
        }
        pick_row.update({c: float(row[c]) for c in ("cd", "emd", "f1") if c in df.columns})
        pick_rows.append(pick_row)

    picks_df = pd.DataFrame(pick_rows)
    picks_df.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")

    abl_tag          = ''.join(ablation_name.strip().lower().split())
    outlier_counters = {}
    for _, row in picks_df.iterrows():
        asset    = row["asset"]
        sel_type = row["selection_type"]
        cd_val   = float(row["cd"])
        f1_val   = float(row["f1"])
        view_id  = int(row["view_id"])

        if sel_type == "outlier":
            outlier_counters[asset] = outlier_counters.get(asset, 0) + 1
            display_sel = f"outlier {outlier_counters[asset]}"
        else:
            display_sel = sel_type

        partial_pcd_path = os.path.join(
            "data", f"NRG_{abl_tag}",
            "projected_partial_noise", asset, asset, "models", f"{view_id}.pcd"
        )
        gt_pcd_path = os.path.join(
            "data", f"NRG_{abl_tag}",
            "NRG_pc", f"{asset}-{asset}-{view_id}.pcd"
        )
        out_graphic = os.path.join(job_out_dir, f"{asset}_{sel_type}_{view_id}_grid.pdf")

        render_triplet_from_pcds(
            partial_pcd_path, gt_pcd_path, out_graphic,
            predictor, asset,
            cd=cd_val, f1=f1_val, sel_type=display_sel,
            include_error=True, panel_size=(512, 512), point_size=2.0,
        )

    return picks_df


# ─────────────────────────────────────────────
# Denoising metrics
# ─────────────────────────────────────────────

def compute_denoising_metrics(partial_norm, complete, gt_norm):
    """
    For each partial point p:
      - partial_to_gt:    dist(p, nearest GT point)
      - completion_to_gt: dist(nearest completion point to p, nearest GT point)

    Returns per-viewpoint summary scalars only (no raw arrays).
    """
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_norm.astype(np.float64))
    gt_tree = o3d.geometry.KDTreeFlann(gt_pcd)

    complete_pcd = o3d.geometry.PointCloud()
    complete_pcd.points = o3d.utility.Vector3dVector(complete.astype(np.float64))
    complete_tree = o3d.geometry.KDTreeFlann(complete_pcd)

    partial_to_gt    = []
    completion_to_gt = []

    for p in partial_norm.astype(np.float64):
        _, _, d2_pg = gt_tree.search_knn_vector_3d(p, 1)
        partial_to_gt.append(np.sqrt(d2_pg[0]))

        _, idx_c, _ = complete_tree.search_knn_vector_3d(p, 1)
        cp = np.asarray(complete_pcd.points)[idx_c[0]]
        _, _, d2_cg = gt_tree.search_knn_vector_3d(cp, 1)
        completion_to_gt.append(np.sqrt(d2_cg[0]))

    partial_to_gt    = np.asarray(partial_to_gt)
    completion_to_gt = np.asarray(completion_to_gt)

    return {
        'partial_to_gt_mean':      float(partial_to_gt.mean()),
        'partial_to_gt_std':       float(partial_to_gt.std()),
        'partial_to_gt_median':    float(np.median(partial_to_gt)),
        'completion_to_gt_mean':   float(completion_to_gt.mean()),
        'completion_to_gt_std':    float(completion_to_gt.std()),
        'completion_to_gt_median': float(np.median(completion_to_gt)),
    }


# ─────────────────────────────────────────────
# Per-ablation inference runner
# ─────────────────────────────────────────────

def run_single_ablation(cfg_path, ckpt_path, test_txt_path, ablation_name, out_path):
    print(f"\n=== Ablation: {ablation_name} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = cfg_from_yaml_file(cfg_path)
    model  = builder.model_builder(config.model)
    builder.load_model(model, ckpt_path)
    model.to(device)
    model.eval()

    predictor = AdaPoinTrPredictor(model, normalize=False)

    with open(test_txt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    file_list = []
    for line in lines:
        taxonomy_id = line.split('-')[0].split('/')[-1]
        model_id    = line.split('-')[1]
        view_id     = int(line.split('-')[2].split('.')[0])
        file_list.append({"taxonomy_id": taxonomy_id, "model_id": model_id,
                           "view_id": view_id, "file_path": line})

    print(f"  {len(file_list)} samples loaded")

    abl_tag = ''.join(ablation_name.strip().lower().split())

    per_sample               = {}
    partial_mean_by_asset    = {}
    completion_mean_by_asset = {}

    for sample in file_list:
        file_path    = os.path.join("data", f"NRG_{abl_tag}", sample['file_path'])
        partial_path = os.path.join("data", f"NRG_{abl_tag}",
                                    "projected_partial_noise",
                                    sample['taxonomy_id'], sample['model_id'],
                                    "models", f"{sample['view_id']}.pcd")

        partial = IO.get(partial_path).astype(np.float32)
        gt0     = IO.get(file_path).astype(np.float32)
        gt, partial_norm, c = _norm_from_partial(gt0, partial)
        complete = predictor.predict(partial_norm)

        denoising = compute_denoising_metrics(partial_norm, complete, gt)

        asset = sample['model_id']
        if asset not in partial_mean_by_asset:
            partial_mean_by_asset[asset]    = []
            completion_mean_by_asset[asset] = []

        partial_mean_by_asset[asset].append(denoising['partial_to_gt_mean'])
        completion_mean_by_asset[asset].append(denoising['completion_to_gt_mean'])

        _metrics = Metrics.get(
            torch.from_numpy(complete).float().cuda().unsqueeze(0),
            torch.from_numpy(gt).float().cuda().unsqueeze(0),
            require_emd=True,
        )

        if asset not in per_sample:
            per_sample[asset] = []

        per_sample[asset].append({
            'idx':                     sample['view_id'],
            'cd':                      2 * _metrics[1],
            'emd':                     2 * _metrics[3],
            'f1':                      _metrics[0],
            'partial_to_gt_mean':      2 * denoising['partial_to_gt_mean'],
            'partial_to_gt_std':       2 * denoising['partial_to_gt_std'],
            'partial_to_gt_median':    2 * denoising['partial_to_gt_median'],
            'completion_to_gt_mean':   2 * denoising['completion_to_gt_mean'],
            'completion_to_gt_std':    2 * denoising['completion_to_gt_std'],
            'completion_to_gt_median': 2 * denoising['completion_to_gt_median'],
        })

    df = per_sample_to_df(per_sample, ablation_name)

    partial_mean_by_asset    = {k: np.array(v, dtype=np.float32) for k, v in partial_mean_by_asset.items()}
    completion_mean_by_asset = {k: np.array(v, dtype=np.float32) for k, v in completion_mean_by_asset.items()}

    # ── Per-asset denoising summary CSV ──────────────────────────────────────
    denoising_rows = []
    for asset in sorted(partial_mean_by_asset.keys()):
        p = partial_mean_by_asset[asset]
        c = completion_mean_by_asset[asset]
        denoising_rows.append({
            "asset":                         asset,
            "ablation":                      ablation_name,
            "n_viewpoints":                  int(len(p)),
            "partial_mean_of_means_mm":      float(p.mean())     * 2,
            "partial_std_of_means_mm":       float(p.std())      * 2,
            "partial_median_of_means_mm":    float(np.median(p)) * 2,
            "completion_mean_of_means_mm":   float(c.mean())     * 2,
            "completion_std_of_means_mm":    float(c.std())      * 2,
            "completion_median_of_means_mm": float(np.median(c)) * 2,
            "denoising_ratio":               float(p.mean()) / max(float(c.mean()), 1e-9),
        })

    denoising_df  = pd.DataFrame(denoising_rows)
    denoising_csv = os.path.join(out_path, f"denoising_{abl_tag}.csv")
    denoising_df.to_csv(denoising_csv, index=False)
    print(f"  Saved {denoising_csv}")

    # ── Per-ablation individual plots ─────────────────────────────────────────
    abl_plot_dir = os.path.join(out_path, "plots", "boxplots", abl_tag)
    save_single_ablation_boxplot(df, "cd",  os.path.join(abl_plot_dir, "cd.pdf"),
                                  title=f"[{ablation_name}] CD by Asset",  ylabel="CD (mm)")
    save_single_ablation_boxplot(df, "f1",  os.path.join(abl_plot_dir, "f1.pdf"),
                                  title=f"[{ablation_name}] F1 by Asset",  ylabel="F1")
    save_single_ablation_boxplot(df, "emd", os.path.join(abl_plot_dir, "emd.pdf"),
                                  title=f"[{ablation_name}] EMD by Asset", ylabel="EMD (mm)")
    save_single_denoising_boxplot(partial_mean_by_asset, completion_mean_by_asset,
                                   os.path.join(abl_plot_dir, "denoising.pdf"))

    # ── Per-ablation graphics (best/median/worst/outliers) ───────────────────
    abl_graphics_dir = os.path.join(out_path, "plots", "graphics", abl_tag)
    df_no_abl = df.drop(columns=["ablation"])
    compute_samples(
        df_no_abl, ablation_name, predictor,
        out_csv=os.path.join(out_path, f"results_{abl_tag}_by_cd.csv"),
        job_out_dir=abl_graphics_dir,
        metric_col="cd",
    )

    df["ablation"] = ablation_name
    return df, partial_mean_by_asset, completion_mean_by_asset


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(ablation_configs, out_path):
    all_dfs        = []
    all_denoising  = []
    ablation_order = [a['name'] for a in ablation_configs]

    for abl in ablation_configs:
        df, partial_means, completion_means = run_single_ablation(
            abl['cfg_path'], abl['ckpt_path'], abl['test_txt_path'],
            abl['name'], out_path,
        )
        all_dfs.append(df)
        all_denoising.append({
            'name':             abl['name'],
            'partial_means':    partial_means,
            'completion_means': completion_means,
        })

    # ── Combined denoising CSV ────────────────────────────────────────────────
    abl_tags = [''.join(a['name'].strip().lower().split()) for a in ablation_configs]
    combined_denoising_df = pd.concat(
        [pd.read_csv(os.path.join(out_path, f"denoising_{tag}.csv")) for tag in abl_tags],
        ignore_index=True,
    )
    combined_denoising_df.to_csv(os.path.join(out_path, "combined_denoising.csv"), index=False)
    print(f"  Saved combined_denoising.csv")

    # ── Combined results CSV ──────────────────────────────────────────────────
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(os.path.join(out_path, "combined_results.csv"), index=False)

    # ── Combined outlier summary + plot ───────────────────────────────────────
    outlier_df = compute_outlier_summary(combined_df, metric="cd")
    outlier_df.to_csv(os.path.join(out_path, "combined_outlier_summary_cd.csv"), index=False)
    print("  Saved combined_outlier_summary_cd.csv")

    combined_plot_dir = os.path.join(out_path, "plots", "boxplots", "combined")

    save_ablation_boxplot(combined_df, "cd",
                           os.path.join(combined_plot_dir, "cd.pdf"),
                           title="Chamfer Distance by Asset and Ablation",
                           ylabel="CD (mm)", ablation_order=ablation_order)
    save_ablation_boxplot(combined_df, "f1",
                           os.path.join(combined_plot_dir, "f1.pdf"),
                           title="F1 Score by Asset and Ablation",
                           ylabel="F1", ablation_order=ablation_order)
    save_ablation_boxplot(combined_df, "emd",
                           os.path.join(combined_plot_dir, "emd.pdf"),
                           title="EMD by Asset and Ablation",
                           ylabel="EMD (mm)", ablation_order=ablation_order)

    save_combined_denoising_boxplot(
        all_denoising,
        os.path.join(combined_plot_dir, "denoising_by_source.pdf"),
        ablation_order=ablation_order,
    )
    save_combined_denoising_by_ablation_boxplot(
        all_denoising,
        os.path.join(combined_plot_dir, "denoising_by_ablation.pdf"),
        ablation_order=ablation_order,
    )

    save_outlier_percentage_barplot(
        outlier_df,
        os.path.join(combined_plot_dir, "outlier_percentage_cd.pdf"),
        metric="cd",
        ablation_order=ablation_order,
    )

    print(f"\nAll done. Results in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AdaPoinTr across multiple ablations and produce combined plots."
    )
    parser.add_argument("--out_path", type=str, default="./results",
                        help="Root output directory")
    parser.add_argument(
        "--ablation", action="append", nargs=4,
        metavar=("NAME", "CFG", "CKPT", "TXT"),
        help=(
            "One ablation: NAME cfg_path ckpt_path test_txt_path. Repeat for each ablation.\n"
            "  --ablation Baseline     cfgs/bl.yaml ckpts/bl.pth test.txt\n"
            "  --ablation 'Ablation 1' cfgs/a1.yaml ckpts/a1.pth test.txt\n"
            "  --ablation 'Ablation 2' cfgs/a2.yaml ckpts/a2.pth test.txt"
        ),
    )
    args = parser.parse_args()

    if not args.ablation:
        parser.error("Provide at least one --ablation NAME CFG CKPT TXT")

    ablation_configs = [
        {"name": a[0], "cfg_path": a[1], "ckpt_path": a[2], "test_txt_path": a[3]}
        for a in args.ablation
    ]

    main(ablation_configs, args.out_path)
