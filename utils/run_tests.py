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
    """
    Build a flat per-viewpoint DataFrame.
    Keys stored in per_sample entries:
        idx, cd, emd, f1,
        partial_to_gt_mean/std/median,
        completion_to_gt_mean/std/median
    """
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
    "font.size":       14,
    "axes.labelsize":  12,
    "axes.titlesize":  16,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
}


def save_single_ablation_boxplot(df, metric, out_path, title, ylabel):
    """Single-ablation boxplot (no hue). One observation per viewpoint."""
    assert metric in df.columns, f"Invalid metric: {metric}"
    plt.rcParams.update(RCPARAMS)

    plot_df = df[np.isfinite(df[metric])].copy()
    fig, ax = plt.subplots(figsize=(4.5, 5.5), constrained_layout=True)
    flierprops = dict(marker='.', markersize=10)
    sns.boxplot(
        data=plot_df, x="asset", y=metric,
        showfliers=True, whis=1.5, ax=ax, flierprops=flierprops,
    )
    ax.set_xlabel('Asset Class', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title or f'{metric.upper()} by Asset', fontsize=14)
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

    # Apply ordering AFTER copying — not before
    if ablation_order is not None:
        plot_df["ablation"] = pd.Categorical(
            plot_df["ablation"], categories=ablation_order, ordered=True
        )

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    flierprops = dict(marker='.', markersize=6, alpha=0.5)
    sns.boxplot(
        data=plot_df, x="asset", y=metric,
        hue="ablation", hue_order=ablation_order,
        showfliers=True, whis=1.5, gap=0.15, width=0.7,
        ax=ax, flierprops=flierprops,
    )

    n_assets = plot_df["asset"].nunique()
    for i in range(n_assets - 1):
        ax.axvline(x=i + 0.5, color='lightgrey', linewidth=1, linestyle='--', zorder=0)

    ax.set_xlabel('Asset Class', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
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
    Same granularity as CD/F1 — eliminates spurious per-point outliers.
    x=Asset, hue=Source (Partial vs Completion).
    """
    records = []
    for asset in sorted(partial_mean_by_asset.keys()):
        for d in partial_mean_by_asset[asset]:       # one float per viewpoint
            records.append({'Asset': asset, 'Source': 'Partial',
                            'Mean Error (mm)': float(d) * 2})
        for d in completion_mean_by_asset[asset]:    # one float per viewpoint
            records.append({'Asset': asset, 'Source': 'Completion',
                            'Mean Error (mm)': float(d) * 2})

    plot_df = pd.DataFrame(records)
    plot_df = plot_df[np.isfinite(plot_df["Mean Error (mm)"])]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sns.boxplot(
        data=plot_df, x="Asset", y="Mean Error (mm)", hue="Source",
        ax=ax, whis=1.5, gap=0.3,
        flierprops=dict(marker='.', markersize=4, alpha=0.5),
    )
    n_assets = plot_df["Asset"].nunique()
    for i in range(n_assets - 1):
        ax.axvline(x=i + 0.5, color='lightgrey', linewidth=1, linestyle='--', zorder=0)

    ax.set_xlabel('Asset Class', fontsize=11)
    ax.set_ylabel('Mean Point-to-GT Distance (mm)', fontsize=11)
    ax.set_title('Partial vs. Completion Denoising Error (per-viewpoint mean)', fontsize=13)
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
    Best for comparing how each ablation affects noise levels within each source type.
    Saved as: combined/denoising_by_source.pdf
    """
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
            flierprops=dict(marker='.', markersize=4, alpha=0.5),
        )
        for i in range(n_assets - 1):
            ax.axvline(x=i + 0.5, color='lightgrey', linewidth=1.0, linestyle='--', zorder=0)
        ax.set_title(f'{source} Point-to-GT Error', fontsize=13)
        ax.set_xlabel('Asset Class', fontsize=11)
        ax.set_ylabel('Mean Error (mm)', fontsize=11)
        ax.legend(title='Ablation', bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=20)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def save_combined_denoising_by_ablation_boxplot(all_denoising, out_path, ablation_order=None):
    """
    Layout B — N panels (one per ablation), each showing x=Asset, hue=Source (Partial vs Completion).
    Best for showing the Partial→Completion denoising improvement within each ablation.
    The gap between the two boxes per asset is the denoising effect.
    Saved as: combined/denoising_by_ablation.pdf
    """
    plot_df = _build_denoising_records(all_denoising)
    plot_df = plot_df[np.isfinite(plot_df["Mean Error (mm)"])]

    abl_list    = ablation_order if ablation_order else sorted(plot_df["Ablation"].unique())
    asset_order = sorted(plot_df["Asset"].unique())
    n_assets    = len(asset_order)
    n_abls      = len(abl_list)

    plot_df["Asset"] = pd.Categorical(
        plot_df["Asset"], categories=asset_order, ordered=True
    )

    # Compute shared y-axis limits so panels are directly comparable
    ymin = 0
    ymax = plot_df["Mean Error (mm)"].quantile(0.99) * 1.1  # clip extreme outliers for scale

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
            flierprops=dict(marker='.', markersize=4, alpha=0.5),
        )
        for i in range(n_assets - 1):
            ax.axvline(x=i + 0.5, color='lightgrey', linewidth=1.0, linestyle='--', zorder=0)

        ax.set_title(abl, fontsize=13, fontweight='bold')
        ax.set_xlabel('Asset Class', fontsize=11)
        ax.set_ylabel('Mean Point-to-GT Error (mm)' if ax == axes[0] else '', fontsize=11)
        ax.set_ylim(ymin, ymax)
        ax.legend(title='Source', bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=20)

    fig.suptitle('Partial vs. Completion Point-to-GT Error by Ablation', fontsize=14)
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
        dists[i] = 1000* np.sqrt(dist2[0]) if len(dist2) > 0 else 0.0
    if vmax is None:
        vmax = 100 # mm
        #vmax = max(1e-6, np.percentile(dists, 95))
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
        title_font   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        caption_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
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
            f"Err. (mean={2*pred_error_stats['mean']:.2f} mm, max={2*pred_error_stats['max']:.2f} mm)"
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
        # Summary scalars — one float per viewpoint
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
    """
    Run inference for one ablation.

    Denoising accumulators store ONE FLOAT PER VIEWPOINT (the per-viewpoint mean
    point-to-GT distance in normalized units). This gives the same observation
    granularity as CD/F1, eliminating spurious per-point outliers.

    Returns:
        df                    — per-viewpoint DataFrame
        partial_mean_by_asset  — {asset: np.array shape (n_viewpoints,)}  [normalized units]
        completion_mean_by_asset — {asset: np.array shape (n_viewpoints,)} [normalized units]
    """
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

    per_sample               = {}  # asset -> list of per-viewpoint dicts
    partial_mean_by_asset    = {}  # asset -> list of floats  (one per viewpoint)
    completion_mean_by_asset = {}  # asset -> list of floats  (one per viewpoint)

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
        # FIX: check against the correct dict name
        if asset not in partial_mean_by_asset:
            partial_mean_by_asset[asset]    = []
            completion_mean_by_asset[asset] = []

        # ONE scalar per viewpoint — not the raw per-point array
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
            # ×2 converts normalized units → mm
            'partial_to_gt_mean':      2 * denoising['partial_to_gt_mean'],
            'partial_to_gt_std':       2 * denoising['partial_to_gt_std'],
            'partial_to_gt_median':    2 * denoising['partial_to_gt_median'],
            'completion_to_gt_mean':   2 * denoising['completion_to_gt_mean'],
            'completion_to_gt_std':    2 * denoising['completion_to_gt_std'],
            'completion_to_gt_median': 2 * denoising['completion_to_gt_median'],
        })

    df = per_sample_to_df(per_sample, ablation_name)

    # FIX: np.array (not np.concatenate) — these are lists of scalar floats
    partial_mean_by_asset    = {k: np.array(v, dtype=np.float32) for k, v in partial_mean_by_asset.items()}
    completion_mean_by_asset = {k: np.array(v, dtype=np.float32) for k, v in completion_mean_by_asset.items()}

    # ── Per-asset denoising summary CSV ──────────────────────────────────────
    # One row per asset. All mm values (×2 applied).
    # Suitable for thesis tables: mean±std and median across viewpoints.
    # denoising_ratio > 1 means completion is closer to GT than partial (good).
    denoising_rows = []
    for asset in sorted(partial_mean_by_asset.keys()):
        p = partial_mean_by_asset[asset]     # (n_viewpoints,) in normalized units
        c = completion_mean_by_asset[asset]  # (n_viewpoints,) in normalized units
        denoising_rows.append({
            "asset":                         asset,
            "ablation":                      ablation_name,
            "n_viewpoints":                  int(len(p)),
            # Partial error across viewpoints (mm)
            "partial_mean_of_means_mm":      float(p.mean())     * 2,
            "partial_std_of_means_mm":       float(p.std())      * 2,
            "partial_median_of_means_mm":    float(np.median(p)) * 2,
            # Completion error across viewpoints (mm)
            "completion_mean_of_means_mm":   float(c.mean())     * 2,
            "completion_std_of_means_mm":    float(c.std())      * 2,
            "completion_median_of_means_mm": float(np.median(c)) * 2,
            # Improvement ratio: partial_mean / completion_mean  (>1 = improvement)
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

    df["ablation"] = ablation_name  # restore for combined df
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
        # Keys must match what save_combined_denoising_boxplot expects
        all_denoising.append({
            'name':             abl['name'],
            'partial_means':    partial_means,
            'completion_means': completion_means,
        })

    # ── Combined denoising CSV (all ablations × all assets — ready for LaTeX tables) ──
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
    # Layout A: 2 panels (Partial | Completion), hue=Ablation — compare ablations within source
    save_combined_denoising_boxplot(
        all_denoising,
        os.path.join(combined_plot_dir, "denoising_by_source.pdf"),
        ablation_order=ablation_order,
    )
    # Layout B: N panels (one per ablation), hue=Source — show Partial->Completion gap per ablation
    save_combined_denoising_by_ablation_boxplot(
        all_denoising,
        os.path.join(combined_plot_dir, "denoising_by_ablation.pdf"),
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






















# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import argparse
# import os
# import numpy as np
# import torch
# import open3d as o3d
# from datasets.io import IO
# from PIL import Image, ImageDraw, ImageFont

# from tools import builder
# from utils.config import cfg_from_yaml_file
# from tools.predictor import AdaPoinTrPredictor
# from extensions.chamfer_dist import ChamferDistanceL1
# from utils.metrics import Metrics
        
# def _norm_from_partial(gt, partial):
#     centroid = np.mean(partial, axis=0)
#     p0 = partial - centroid
    
#     partial0 = p0 / 2.0
#     gt0 = (gt - centroid) / 2.0

#     return gt0.astype(np.float32), partial0.astype(np.float32), centroid

# def per_sample_to_df(per_sample):
#     rows = []
#     for asset, entries in per_sample.items():
#         for e in entries:
#             rows.append({
#                 "asset": asset,
#                 "view_id": e["idx"],
#                 "cd": float(e["cd"]),
#                 "emd": float(e["emd"]),
#                 "f1": float(e["f1"]),
#                 "partial_to_gt_mean": float(e["partial_to_gt_mean"]),
#                 "partial_to_gt_std": float(e["partial_to_gt_std"]),
#                 "partial_to_gt_median": float(e["partial_to_gt_median"]),
#                 "partial_to_completion_mean": float(e["partial_to_completion_mean"]),
#                 "partial_to_completion_std": float(e["partial_to_completion_std"]),
#                 "partial_to_completion_median": float(e["partial_to_completion_median"]),
#             })
            
#     df = pd.DataFrame(rows)
#     df["asset"] = pd.Categorical(df["asset"], 
#                                  categories=sorted(df["asset"].unique()),
#                                  ordered=True)
    
#     return df

# def save_denoising_boxplot(partial_raw_by_asset, completion_raw_by_asset, out_path):
#     records = []

#     for asset in sorted(partial_raw_by_asset.keys()):
#         for d in partial_raw_by_asset[asset]:
#             records.append({'Asset': "asset", 'Source': 'Partial', 'Error (mm)': float(d) * 2})
#         for d in completion_raw_by_asset[asset]:
#             records.append({'Asset': "asset", 'Source': 'Completion', 'Error (mm)': float(d) * 2})

#     plot_df = pd.DataFrame(records)
#     plot_df = plot_df[np.isfinite(plot_df["Error (mm)"])]

#     fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
#     sns.boxplot(data=plot_df, x="Asset", y="Error (mm)", hue="Source", ax=ax, whis=1.5, gap=0.3,
#                 flierprops=dict(marker='.', markersize=4, alpha=0.5))


#     ax.set_xlabel('Asset Class', fontsize=11)
#     ax.set_ylabel('Point-to-GT Distance (mm)', fontsize=11)
#     ax.set_title('Partial vs. Completion Error in Observed Region', fontsize=13)

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.savefig(out_path, dpi=600)
#     plt.close()

# def save_boxplot(df, metric, out_path, title, ylabel):
#     assert metric in df.columns, f"Invalid metric: {metric}"
#     fig, ax = plt.subplots(figsize=(4.5, 5.5), constrained_layout=True)
#     fig.subplots_adjust(bottom=0.20)

#     plot_df = df[np.isfinite(df[metric])].copy()
#     plt.rcParams.update({
#         "font.size": 14,
#         "axes.labelsize": 12,
#         "axes.titlesize": 16,
#         "xtick.labelsize": 8,
#         "ytick.labelsize": 8,
#         "legend.fontsize": 12})

#     #plt.figure(figsize=(4.5, 5.5))
#     flierprops = dict(
#         marker='.',
#         markersize=10)
#     sns.boxplot(
#         data=plot_df,
#         x="asset",
#         y=metric,
#         showfliers=True,
#         whis=1.5, ax=ax, flierprops=flierprops
#     )

#     ax.set_xlabel('Asset Class', fontsize=11)
#     ax.set_ylabel(ylabel, fontsize=11)
#     if title is None:
#         title = f'{metric.upper()} distribution across viewpoints (per asset)'
#     ax.set_title(title, fontsize=14)

#     plt.xticks(rotation=20, ha="right")
#     #plt.tight_layout()

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
#     plt.savefig(out_path, dpi=600, bbox_inches=None)
#     plt.close()

# def _compute_error_colormap(pred, gt, cmap='viridis', vmax=None):
#     '''Return Nx3 colors for pred ponts = nearest distance to GT mapped to colormap'''
#     pred_np =  np.asarray(pred.points)
#     gt_np = np.asarray(gt.points)

#     if pred_np.size == 0 or gt_np.size == 0:
#         return np.zeros((pred_np.shape[0], 3), dtype=np.float32), np.zeros((pred_np.shape[0],), dtype=np.float32)
    
#     tree = o3d.geometry.KDTreeFlann(pred) #o3d.geometry.KDTreeFlann(gt)

#     dists = np.zeros((gt_np.shape[0],), dtype=np.float32)

#     for i, p in enumerate(gt_np): #enumerate(pred_np):
#         _, idx, dist2 = tree.search_knn_vector_3d(p, 1)
#         dists[i] = np.sqrt(dist2[0]) if len(dist2) > 0 else 0.0

#     if vmax is None:
#         vmax = max(1e-6, np.percentile(dists, 95))

#     norm = np.clip(dists / float(vmax), 0.0, 1.0)
#     cmap_func = plt.get_cmap(cmap)
#     colors = cmap_func(norm)[:, :3]
#     return colors.astype(np.float64), dists

# def _render_pcd_to_image(pcd,
#                          center,
#                          cam_pos,
#                          cam_up, 
#                          radius, 
#                          distance,
#                          width=512,
#                          height=512,
#                          point_size=3.5,
#                          visible=False
#                          ):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=width, height=height, visible=visible)
#     vis.add_geometry(pcd)

#     ctr = vis.get_view_control()
    
#     front = (center - cam_pos).astype(np.float64)
#     norm = np.linalg.norm(front) + 1e-12
#     front = front / norm

#     ctr.set_lookat(center.tolist())
#     ctr.set_front(front.tolist())
#     ctr.set_up(cam_up.tolist())

#     zoom = 0.85 * (radius / distance)
#     ctr.set_zoom(float(zoom))

#     opt = vis.get_render_option()
#     opt.background_color = np.asarray([1.0, 1.0, 1.0])
#     opt.point_size = float(point_size)

#     vis.update_geometry(pcd)

#     vis.poll_events()
#     vis.update_renderer()

#     img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
#     vis.destroy_window()

#     img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

#     return img8 

# def _make_camera_params_from_gt(gt_pcd, cam_offset_factor=(0.8, -1.0, 1.6),
#                                 fov_deg=60.0,
#                                 width=512, height=512,
#                                 point_size=3.5,
#                                 visible=False):
#     """
#     Create a PinholeCameraParameters sampled from a GT visualizer. This returns
#     a PinholeCameraParameters object that can be reused on other Visualizers so
#     they use identical extrinsic+intrinsic.
#     """
#     # compute bbox anchor
#     bbox = gt_pcd.get_axis_aligned_bounding_box()
#     center = bbox.get_center()
#     extent = bbox.get_extent()
#     radius = float(np.linalg.norm(extent) * 0.5)
#     radius = radius if radius > 0 else 1.0

#     cam_offset = np.array([cam_offset_factor[0] * radius,
#                            cam_offset_factor[1] * radius,
#                            cam_offset_factor[2] * radius], dtype=np.float64)
    

#     z = 2*radius
#     xy = np.array([1.0, 0.0], dtype=np.float64) * radius
#     yaw_deg=45.0
#     if abs(yaw_deg) > 1e-6:
#         th = np.deg2rad(yaw_deg)
#         Rz = np.array([[np.cos(th), -np.sin(th)],
#                        [np.sin(th), np.cos(th)]], dtype=np.float64)
#         xy = Rz.dot(xy)
    


#     cam_pos = center + np.array([xy[0], xy[1], z], dtype=np.float64) #cam_offset
#     cam_up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)

#     # Create a short-lived visualizer with the GT cloud to set a camera
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=width, height=height, visible=visible)
#     vis.add_geometry(gt_pcd)

#     ctr = vis.get_view_control()

#     # compute normalized front vector (camera -> center)
#     front = (center - cam_pos).astype(np.float64)
#     front /= (np.linalg.norm(front) + 1e-12)

#     # Set the camera explicitly
#     ctr.set_lookat(center.tolist())
#     ctr.set_front(front.tolist())
#     ctr.set_up(cam_up.tolist())
#     print(f"Camera set up to {cam_up.tolist()}")
#     # optionally tune zoom a bit; this is still needed to get proper scale
#     ctr.set_zoom(0.75)

#     # convert to pinhole parameters (this captures the extrinsic matrix used)
#     params = ctr.convert_to_pinhole_camera_parameters()

#     # Replace intrinsics with a controlled intrinsic using a simple fov model
#     # compute fx/fy from requested fov and image width
#     fov_rad = np.deg2rad(float(fov_deg))
#     fx = fy = 0.5 * width / np.tan(0.5 * fov_rad)
#     cx = width * 0.5
#     cy = height * 0.5
   
#     intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
#     params.intrinsic = intr
#     params.intrinsic.width = int(width)
#     params.intrinsic.height = int(height)
#     # clean up
#     vis.destroy_window()

#     # Return anchor info so caller can reuse center/radius if needed
#     anchor = {"params": params, "center": center, "radius": radius, "cam_pos": cam_pos, "cam_up": cam_up}
#     return anchor

# def _render_with_camera_params(pcd, params, width=512, height=512, point_size=3.5, visible=False):
#     """
#     Render a single point cloud using the provided PinholeCameraParameters.
#     """
#     vis = o3d.visualization.Visualizer()
#     win_w = int(getattr(params.intrinsic, "width", 0))
#     win_h = int(getattr(params.intrinsic, "height", 0))
#     vis.create_window(width=win_w, height=win_h, visible=visible)
#     vis.add_geometry(pcd)

#     print(f"[DEBUG] Render window size = ({win_w} {win_h}), params.intrinsic size = ({params.intrinsic.width} {params.intrinsic.height})")

#     ctr = vis.get_view_control()
#     # apply the previously-captured parameters exactly
#     try:
#         print("CONVERT...")
#         ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
#     except Exception:
#         # fallback: if convert_from fails for some Open3D installs, try set_lookat/front/up
#         cam = params.extrinsic
#         # still attempt approximate fallback
#         # (we expect convert_from_pinhole_camera_parameters to work in most setups)
#         pass

#     opt = vis.get_render_option()
#     opt.background_color = np.asarray([1.0, 1.0, 1.0])
#     opt.point_size = float(point_size)

#     vis.update_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()
#     img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
#     vis.destroy_window()
#     return (np.clip(img, 0, 1) * 255).astype(np.uint8)

# def render_triplet_from_pcds(partial_pcd_path,
#                              gt_pcd_path,
#                              out_path,
#                              predictor,
#                              asset,
#                              cd,
#                              f1,
#                              sel_type,
#                              include_error=True,
#                              panel_size=(512, 512),
#                              point_size=3.0,
#                              title_font_path=None
#                              ):
    
#     assert os.path.exists(partial_pcd_path), f"Partial PCD {partial_pcd_path} not found"
#     assert os.path.exists(gt_pcd_path), f"GT PCD {gt_pcd_path} not found"

#     # Load PCDs
#     partial_pcd = o3d.io.read_point_cloud(partial_pcd_path)
#     gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
    
#     partial_pts = np.asarray(partial_pcd.points)
#     gt_pts = np.asarray(gt_pcd.points)

#     print("PARTIAL: n_pts", partial_pts.shape[0],
#       "centroid", partial_pts.mean(axis=0) if partial_pts.size else None,
#       "extent", (partial_pts.max(axis=0)-partial_pts.min(axis=0)) if partial_pts.size else None)
 
#     print("GT:      n_pts", gt_pts.shape[0],
#       "centroid", gt_pts.mean(axis=0) if gt_pts.size else None,
#       "extent", (gt_pts.max(axis=0)-gt_pts.min(axis=0)) if gt_pts.size else None)
    
#     # Compute prediction
#     input = IO.get(partial_pcd_path).astype(np.float32)
#     gt_norm = IO.get(gt_pcd_path).astype(np.float32)

#     gt_norm, input, c = _norm_from_partial(gt_norm, input)
#     complete = predictor.predict(input)

#     # Denormalize
#     complete = complete * 2.0 + c

#     # Convert to open3d pcd
#     complete_pcd = o3d.geometry.PointCloud()
#     complete_pcd.points = o3d.utility.Vector3dVector(complete)
#     complete_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (complete.shape[0], 1)))

#     # # Compute camera anchor
#     # gt_np = np.asarray(gt_pcd.points)
#     # if gt_np.size == 0:
#     #     center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
#     #     radius = 1.0
#     # else:
#     #     center = gt_np.mean(axis=0)
#     #     radius = float(np.max(np.linalg.norm(gt_np - center, axis=1)))
#     #     radius = radius if radius > 0 else 1.0
    
#     # w, h = panel_size

#     # Align partial and completion to gt centroid
#     # partial_cent = partial_pts.mean(axis=0)
#     # complete_cent = complete.mean(axis=0)
#     # gt_cent = gt_pts.mean(axis=0)

#     # partial_pcd = partial_pcd.translate(gt_cent - partial_cent)
#     # complete_pcd = complete_pcd.translate(gt_cent - complete_cent)

#     # bbox = gt_pcd.get_axis_aligned_bounding_box()
#     # center = bbox.get_center()
#     # extent = bbox.get_extent()
#     # radius = float(np.linalg.norm(extent) * 0.5)
#     # distance = 1.1 * radius 

#     # cam_offset = np.array([0.8 * radius, -1.0 * radius, 1.6 * radius], dtype=np.float64)
#     # cam_pos = center + cam_offset
#     # cam_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
#     # img_partial = _render_pcd_to_image(partial_pcd, center, cam_pos, cam_up, radius, distance, width=w, height=h, point_size=point_size)
#     # img_gt = _render_pcd_to_image(gt_pcd, center, cam_pos, cam_up, radius, distance, width=w, height=h, point_size=point_size)
#     # img_complete = _render_pcd_to_image(complete_pcd, center, cam_pos, cam_up, radius, distance, width=w, height=h, point_size=point_size)

#     # panels = [Image.fromarray(img_partial), Image.fromarray(img_complete), Image.fromarray(img_gt)]
#     bbox = gt_pcd.get_axis_aligned_bounding_box()
#     center = bbox.get_center()
#     extent = bbox.get_extent()
#     radius = float(np.linalg.norm(extent) * 0.5)
#     radius = radius if radius > 0 else 1.0
#     distance = 1.5 * radius

#     # pick a 3/4 top view (positive Z so you see the top)
#     cam_offset = np.array([0.8 * radius, -1.0 * radius, 1.6 * radius], dtype=np.float64)
#     cam_pos = center + cam_offset
#     cam_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

#     # Debug prints (optional)
#     print("CAM anchor center:", center, "radius:", radius)
#     print("cam_pos:", cam_pos, "cam_up:", cam_up)

#     # Render helper that explicitly sets camera parameters (do not change the cloud)
#     # def _render_pcd_with_fixed_camera(pcd, center, cam_pos, cam_up, width=512, height=512, point_size=3.5, visible=False):
#     #     vis = o3d.visualization.Visualizer()
#     #     vis.create_window(width=width, height=height, visible=visible)
#     #     vis.add_geometry(pcd)
#     #     ctr = vis.get_view_control()

#     #     # compute normalized front vector from camera -> center
#     #     front = (center - cam_pos).astype(np.float64)
#     #     front /= (np.linalg.norm(front) + 1e-12)

#     #     # Set camera explicitly — this makes the view deterministic across pcds
#     #     try:
#     #         ctr.set_lookat(center.tolist())
#     #         ctr.set_front(front.tolist())
#     #         ctr.set_up(cam_up.tolist())
#     #     except Exception:
#     #         # older Open3D versions sometimes behave differently — ignore and keep going
#     #         pass

#     #     # deterministic zoom; tweak 0.6-0.85 depending how tightly you want to frame
#     #     ctr.set_zoom(0.75)

#     #     opt = vis.get_render_option()
#     #     opt.background_color = np.asarray([1.0, 1.0, 1.0])
#     #     opt.point_size = float(point_size)

#     #     vis.update_geometry(pcd)
#     #     vis.poll_events()
#     #     vis.update_renderer()
#     #     img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
#     #     vis.destroy_window()
#     #     return (np.clip(img, 0, 1) * 255).astype(np.uint8)

#     # Use the function on the raw point clouds (no translation)
#     w, h = panel_size
#     anchor = _make_camera_params_from_gt(gt_pcd, cam_offset_factor=(0.8, -1.0, 1.6),
#                                         fov_deg=60.0, width=w, height=h,
#                                         point_size=point_size, visible=False)
#     params = anchor['params']

#     # 2) render each cloud using the exact same params (no translations)
#     img_partial  = _render_with_camera_params(partial_pcd,  params, width=w, height=h, point_size=point_size, visible=False)
#     img_complete = _render_with_camera_params(complete_pcd, params, width=w, height=h, point_size=point_size, visible=False)
#     img_gt       = _render_with_camera_params(gt_pcd,       params, width=w, height=h, point_size=point_size, visible=False)

#     # Now compose panels as you already do
#     panels = [Image.fromarray(img_partial), Image.fromarray(img_complete), Image.fromarray(img_gt)]
#     # img_partial  = _render_pcd_with_fixed_camera(partial_pcd,  center, cam_pos, cam_up, width=w, height=h, point_size=point_size)
#     # img_complete = _render_pcd_with_fixed_camera(complete_pcd, center, cam_pos, cam_up, width=w, height=h, point_size=point_size)
#     # img_gt       = _render_pcd_with_fixed_camera(gt_pcd,       center, cam_pos, cam_up, width=w, height=h, point_size=point_size)

#     # Then combine panels exactly like you already do
#     panels = [Image.fromarray(img_partial), Image.fromarray(img_complete), Image.fromarray(img_gt)]

#     pred_error_stats = None
#     if include_error: 
#         err_colors, dists = _compute_error_colormap(complete_pcd, gt_pcd, cmap='viridis', vmax=None)
        
#         #complete_pts = np.asarray(complete_pcd.points)
#         pred_err_pcd = o3d.geometry.PointCloud()
#         pred_err_pcd.points = o3d.utility.Vector3dVector(gt_pts.astype(np.float64))
#         pred_err_pcd.colors = o3d.utility.Vector3dVector(err_colors)
#         img_err = _render_with_camera_params(pred_err_pcd, params, width=w, height=h, point_size=3.5, visible=False)
#         panels.append(Image.fromarray(img_err))
#         pred_error_stats = {"mean": dists.mean(), "max": float(np.max(dists)), "min": float(np.min(dists))}
#     pad_between = 0
#     grid_w = 2 * w + pad_between
#     grid_h = 2 * h + pad_between
#     #num_panels = len(panels)
#     #total_w = w * num_panels

#     try: 
#         if title_font_path and os.path.exists(title_font_path):
#             title_font = ImageFont.truetype(title_font_path,  20)
#             caption_font  = ImageFont.truetype(title_font_path,  15)
#         else:
#             try:
#                 title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  20)
#                 caption_font  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  15)
#             except Exception:
#                 title_font = ImageFont.load_default()
#                 caption_font = ImageFont.load_default()
#     except Exception:
#         title_font = ImageFont.load_default()
#         caption_font = ImageFont.load_default()

#     padding = 0

#     title_height = int(5)
#     caption_height = int(30)
#     footer_padding = 30
#     final_w = grid_w
#     final_h = title_height + grid_h + caption_height + footer_padding
#     out_image = Image.new("RGB", (final_w, final_h), (255, 255, 255))
#     draw = ImageDraw.Draw(out_image)

#     y_panels = title_height
#     #out_image.paste(panels[0], (0, y_panels))
#     #out_image.paste(panels[1], (w, y_panels))
#     #out_image.paste(panels[3], (0, y_panels + h))
#     #out_image.paste(panels[2], (w, y_panels + h))
#    # for i, p in enumerate(panels):
#     #    out_image.paste(p, (i * w, y_panels))

#     # Title text
#     cd_txt = f"CD: {cd:.2f} mm" if cd is not None else "CD: N/A"
#     f1_txt = f"F1: {f1:.2f}" if f1 is not None else "F1: N/A"
#     if sel_type == "lowest":
#         sel_type = "Best"
#     elif sel_type == "median":
#         sel_type = "Median"
#     elif sel_type == "highest":
#         sel_type = "Worst"
#     elif "outlier" in sel_type:
#         sel_type = sel_type.capitalize()
#     else:
#         sel_type = "Unknown"

#     title_txt = f"{asset.capitalize()} {sel_type.capitalize()} ({cd_txt}, {f1_txt})"
#     tmp_img = Image.new("RGB", (10, 10))
#     tmp_draw = ImageDraw.Draw(tmp_img)
#     bbox_title = tmp_draw.textbbox((0, 0), title_txt, font=title_font)

#     title_text_h = bbox_title[3] - bbox_title[1]
#     title_pad_top = 4
#     title_pad_bottom =  6
#     title_height = title_text_h + title_pad_top + title_pad_bottom
#     final_h = title_height + grid_h + caption_height + footer_padding
#     out_image = Image.new("RGB", (final_w, final_h), (255, 255, 255))
#     draw = ImageDraw.Draw(out_image)

#     y_panels = title_height
#     out_image.paste(panels[0], (0, y_panels))
#     out_image.paste(panels[1], (w, y_panels))
#     out_image.paste(panels[3], (0, y_panels + h))
#     out_image.paste(panels[2], (w, y_panels + h))
#     bbox = draw.textbbox((0, 0), title_txt, font=title_font)
#     title_w = bbox[2] - bbox[0]
#     title_h = bbox[3] - bbox[1]
#     title_x = max(0, (final_w - title_w) // 2)
#     title_y = title_pad_top #max(4, (title_height - title_h) // 2)
#     draw.text((title_x, title_y), title_txt, fill=(0, 0, 0), font=title_font)

#     caption_labels = ["Part.", "Pred."]
#     if include_error:
#         caption_labels.append(f"Err. (mean={2*pred_error_stats['mean']:.2f} mm, max={2 * pred_error_stats['max']:.2f} mm)")
#     caption_labels.append("GT")
#     #cap_y = y_panels + grid_h + (caption_height - 12) // 2
#     for i, label in enumerate(caption_labels):
#         row = i // 2
#         col = i % 2

#         cell_x = col * (w + pad_between)
#         cell_y_top = y_panels + row * (h + pad_between)

#         bbox = draw.textbbox((0, 0), label, font=caption_font)
#         cap_w = bbox[2] - bbox[0]
#         cap_h = bbox[3] - bbox[1]
        
#         cap_x = cell_x + (w - cap_w) // 2 #i * w + (w - cap_w) // 2
#         cap_y = cell_y_top + h + max(2, (caption_height - cap_h) // 2)
#         draw.text((cap_x, cap_y), label, fill=(0, 0, 0), font=caption_font)


#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     out_image.save(out_path)

#     info = {
#         "center": center.tolist(),
#         "radius": radius,
#         "cam_pos": cam_pos.tolist(),
#         "panels": len(panels),
#         "pred_error": pred_error_stats
#     }

#     print(f"Render saved {out_path}")

#     return out_path, info

# def compute_samples(df, ablation,
#                     predictor,
#                     out_csv="selected_samples.csv",
#                     job_out_dir="metrics",
#                     metric_col="cd"
#                     ):
#     # For CD:
#     # Get lowest performing
#     # Get median
#     # Get highest performing sample (non-outlier)
#     # Get all outliers
#     assert metric_col in df.columns, f"Metric {metric_col} not found"
#     picks = []

#     grouped = df.groupby("asset")

#     for asset, g in grouped:
#         vals = g[metric_col].dropna().values
#         if len(vals) == 0:
#             continue

#         q1 = np.percentile(vals, 25)
#         q2 = np.percentile(vals, 50)
#         q3 = np.percentile(vals, 75)
#         iqr = q3 - q1

#         outlier_high = q3 + 1.5 * iqr
#         outlier_low = q1 - 1.5 * iqr

#         # All outliers
#         # Highest non-outlier
#         non_outlier_mask = (g[metric_col] >= outlier_low) & (g[metric_col] <= outlier_high)
#         outlier_mask = (g[metric_col] < outlier_low) | (g[metric_col] > outlier_high)

#         non_outlier_df = g[non_outlier_mask].copy()
#         outlier_df = g[outlier_mask].copy()

#         if len(non_outlier_df) == 0:
#             non_outlier_df = g.copy()

#         highest_non_outlier = float(non_outlier_df[metric_col].max())
#         lowest_non_outlier = float(non_outlier_df[metric_col].min())

#         # Find the row
#         def find_closest_row(value, kind):
#             idx = (g[metric_col] - value).abs().idxmin()
#             row = g.loc[idx]
#             return idx, row
        
#         try: 
#             idx_min, row_min = find_closest_row(lowest_non_outlier, "lowest")
#             picks.append((asset, metric_col, "lowest", idx_min, row_min))
#         except: 
#             print(f"[WARN] Couldn't pick lowest non-outlier for {asset}")

#         try: 
#             idx_med, row_med = find_closest_row(q2, "median")
#             picks.append((asset, metric_col, "median", idx_med, row_med))
#         except:
#             print(f"[WARN] Couldn't pick median for {asset}")

#         try: 
#             idx_max, row_max = find_closest_row(highest_non_outlier, "highest")
#             picks.append((asset, metric_col, "highest", idx_max, row_max))
#         except:
#             print(f"[WARN] Couldn't pick highest non-outlier for {asset}")

#         # Outliers
#         for oi, orow in outlier_df.iterrows():
#             picks.append((asset, metric_col, "outlier", oi, orow))
        
#     # Build dataframe
#     pick_rows = []
#     for asset, metric_used, sel_type, idx, row in picks:
#         all_metrics = {c: float(row[c]) for c in df.columns if c in ("cd", "emd", "f1")}

#         pick_row = {
#             "asset": asset,
#             "selection_type": sel_type,
#             "selection_metric": metric_used,
#             "df_index": int(idx),
#             "view_id": int(row["view_id"]),
#         }

#         # Merge metric values
#         pick_row.update(all_metrics)
#         pick_rows.append(pick_row)

#     picks_df = pd.DataFrame(pick_rows)

#     # Save CSV
#     picks_df.to_csv(out_csv, index=False)
#     print(f"[RESULT] Saved {out_csv}")

#     # Render and save as: asset_grid
#     grouped = picks_df.groupby("asset")
#     for asset, g in grouped:
#         outlier_i = 1
#         for _, row in g.iterrows():
#             sel_type = row["selection_type"]
#             cd = float(row["cd"])
#             f1 = float(row["f1"])
#             view_id = int(row["view_id"])

#             partial_pcd_path = os.path.join("data", f"NRG_{ablation}", "projected_partial_noise", asset, asset, "models", f"{view_id}.pcd")
#             gt_pcd_path = os.path.join("data", f"NRG_{ablation}", "NRG_pc", f"{asset}-{asset}-{view_id}.pcd")

#             out_path = os.path.join(job_out_dir, f"{asset}_{sel_type}_{view_id}_grid.pdf")
#             if sel_type=="outlier":
#                 sel_type = f"outlier {outlier_i}"
#                 outlier_i = outlier_i + 1

#             render_triplet_from_pcds(partial_pcd_path,
#                                  gt_pcd_path,
#                                  out_path,
#                                  predictor,
#                                  asset,
#                                  cd=cd,
#                                  f1=f1,
#                                  sel_type=sel_type,
#                                  include_error=True,
#                                  panel_size=(512, 512),
#                                  point_size=2.0,)
#     return picks_df

# def compute_denoising_metrics(partial_norm,
#                               complete,
#                               gt_norm
#     ):
#     gt_pcd = o3d.geometry.PointCloud()
#     gt_pcd.points = o3d.utility.Vector3dVector(gt_norm.astype(np.float64))
#     gt_tree = o3d.geometry.KDTreeFlann(gt_pcd)

#     complete_pcd = o3d.geometry.PointCloud()
#     complete_pcd.points = o3d.utility.Vector3dVector(complete.astype(np.float64))
#     complete_tree = o3d.geometry.KDTreeFlann(complete_pcd)

#     partial_to_gt = []
#     completion_to_gt = []

#     for p in partial_norm.astype(np.float64):
#         _, _, dist2_partial_gt = gt_tree.search_knn_vector_3d(p, 1)
#         partial_to_gt.append(np.sqrt(dist2_partial_gt[0]))

#         _, idx_complete, _ = complete_tree.search_knn_vector_3d(p, 1)
#         corresponding_complete_pt = np.asarray(complete_pcd.points)[idx_complete[0]]

#         _, _, dist2_completion_gt = gt_tree.search_knn_vector_3d(corresponding_complete_pt, 1)
#         completion_to_gt.append(np.sqrt(dist2_completion_gt[0]))

#     partial_to_gt = np.asarray(partial_to_gt)
#     completion_to_gt = np.asarray(completion_to_gt)

#     return {
#         'partial_to_gt_mean': float(partial_to_gt.mean()),
#         'partial_to_gt_std': float(partial_to_gt.std()),
#         'partial_to_gt_median': float(np.median(partial_to_gt)),
#         'completion_to_gt_mean': float(completion_to_gt.mean()),
#         'completion_to_gt_std': float(completion_to_gt.std()),
#         'completion_to_gt_median': float(np.median(completion_to_gt)),
#         'partial_to_gt_raw': partial_to_gt,
#         'completion_to_gt_raw': completion_to_gt
#     }


# def main(cfg_path,
#          ckpt_path,
#          test_txt_path,
#          out_path,
#          ablation
#          ):

#     assert os.path.exists(cfg_path), "Config file missing"
#     assert os.path.exists(ckpt_path), "Checkpoint missing"


#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # # ---- build model ----
#     config = cfg_from_yaml_file(cfg_path)
#     model = builder.model_builder(config.model)
#     builder.load_model(model, ckpt_path)
#     model.to(device)
#     model.eval()

#     # # ---- predictor wrapper ----
#     predictor = AdaPoinTrPredictor(model, normalize=False)

#     # Loop through each taxonomy and model, and run inference, computing F1 and CD, and plotting distribution, and also saving 10th, 50th, and 90th percentiles
#     with open(test_txt_path, "r") as f:
#         lines = [ln.strip() for ln in f.readlines() if ln.strip()]

#     file_list = []
#     for line in lines:
#         # taxonomy-model-00042
#         line = line.strip()
#         taxonomy_id = line.split('-')[0].split('/')[-1]
#         model_id = line.split('-')[1]
#         view_id = int(line.split('-')[2].split('.')[0])


#         file_list.append({
#             "taxonomy_id": taxonomy_id,
#             "model_id": model_id,
#             "view_id": view_id,
#             "file_path": line,
#         })

#     print(f"[DATASET] {len(file_list)} samples were loaded")

#     per_sample = {}
#     partial_raw_by_asset = {}
#     completion_raw_by_asset = {}

#     for sample in file_list:
#         print(f"Processing {sample['file_path']}...")
#         # Open partial PCD and run inference
#         # file_path: NRG_pc/glovebox-glovebox-215.pcd
#         # partial_path: projected_partial_noise/glovebox/glovebox/models/215.pcd
#         file_path = os.path.join("data", f"NRG_{ablation}", sample['file_path'])
#         partial_path = os.path.join("data", f"NRG_{ablation}", "projected_partial_noise", sample['taxonomy_id'], sample['model_id'], "models", f"{sample['view_id']}.pcd")
   
#         partial = IO.get(partial_path).astype(np.float32)
#         gt0 = IO.get(file_path).astype(np.float32)

#         print(f"partial {partial_path}, gt {file_path}")
#         gt, partial_norm, c = _norm_from_partial(gt0, partial)
        
#         # Predictor returns points normalized
#         complete = predictor.predict(partial_norm)

#         denoising = compute_denoising_metrics(partial_norm, complete, gt)

#         asset = sample['model_id']
#         if asset not in partial_raw_by_asset.keys():
#             partial_raw_by_asset[asset] = []
#             completion_raw_by_asset[asset] = []

#         partial_raw_by_asset[asset].append(denoising['partial_to_gt_raw'])
#         partial_raw_by_asset[asset].append(denoising['completion_to_gt_raw'])


#         # Compute CD and F1 between complete and gt
#         _metrics = Metrics.get(torch.from_numpy(complete).float().cuda().unsqueeze(0), torch.from_numpy(gt).float().cuda().unsqueeze(0), require_emd=True)
        
#         if sample['model_id'] not in per_sample.keys():
#             per_sample[sample['model_id']] = []

#         per_sample[sample['model_id']].append({
#              'idx': sample['view_id'], 
#              'cd': 2 * _metrics[1],
#              'emd': 2 * _metrics[3],
#              'f1': _metrics[0],
#              'partial_to_gt_mean': 2*denoising['partial_to_gt_mean'],
#              'partial_to_gt_std': 2*denoising['partial_to_gt_std'],
#              'partial_to_gt_median': 2*denoising['partial_to_gt_median'],
#              'partial_to_completion_median': 2*denoising['partial_to_completion_median'],
#              'partial_to_completion_std': 2*denoising['partial_to_completion_std'],
#              'partial_to_completion_mean': 2*denoising['partial_to_completion_mean'],
#              #'metrics': [float(x) for x in _metrics]
#          })
        
#     # Run analysis
#     df = per_sample_to_df(per_sample)

#     # Plot
#     save_boxplot(df, "cd", os.path.join(out_path, "plots/boxplots/test_cd_boxplot.pdf"), title="Test Set CD by Asset", ylabel="CD (mm)")
#     save_boxplot(df, "emd", os.path.join(out_path, "plots/boxplots/test_emd_boxplot.pdf"), title="Test Set EMD by Asset", ylabel="EMD (mm)")
#     save_boxplot(df, "f1", os.path.join(out_path, "plots/boxplots/test_f1_boxplot.pdf"), title="Test Set F1 Score by Asset", ylabel="F1")



#     partial_raw_by_asset = {k: np.concatenate(v) for k, v in partial_raw_by_asset.items()}
#     completion_raw_by_asset = {k: np.concatenate(v) for k, v in completion_raw_by_asset.items()}

#     save_denoising_boxplot(partial_raw_by_asset, completion_raw_by_asset, 
#                            os.path.join(out_path, "plots/boxplots/test_denoising_boxplot.pdf"))
#     # get iqs
#     compute_samples(df, ablation, predictor,
#                     os.path.join(out_path, "results_by_cd.csv"),
#                     os.path.join(out_path, "plots/graphics"),
#                     metric_col="cd")
    
#     #compute_samples(df, predictor,
#      #               os.path.join(out_path, "results_by_f1.csv"),
#       #              os.path.join(out_path, "plots/graphics"),
#        #             metric_col="f1")
    
#    # compute_samples(df, predictor,
#     #            os.path.join(out_path, "results_by_emd.csv"),
#      #           os.path.join(out_path, "plots/graphics"),
#       #          metric_col="emd")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--cfg_path",
#         required= True, 
#         type = str, 
#         help = 'Config file path')
#     parser.add_argument(
#         "--ckpt_path", 
#         required = True,
#         type = str, 
#         help = 'Checkpoint file path')
#     parser.add_argument(
#         "--test_txt_path",
#         required = True, 
#         type = str, 
#         help = 'Path to txt file containing list of test PCDs')
#     parser.add_argument(
#         "--out_path",
#         type = str, 
#         default="./results",
#         help = 'Output path')
#     parser.add_argument(
#         "--ablation",
#         type=str,
#         default="baseline",
#         help = "Ablation")
#     args = parser.parse_args()

#     main(cfg_path=args.cfg_path, ckpt_path=args.ckpt_path, test_txt_path=args.test_txt_path, out_path=args.out_path, ablation=args.ablation)
