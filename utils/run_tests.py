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
from extensions.chamfer_dist import ChamferDistanceL1
from utils.metrics import Metrics
        
def _norm_from_partial(gt, partial):
    centroid = np.mean(partial, axis=0)
    p0 = partial - centroid
    
    partial0 = p0 / 2.0
    gt0 = (gt - centroid) / 2.0

    return gt0.astype(np.float32), partial0.astype(np.float32), centroid

def per_sample_to_df(per_sample):
    rows = []
    for asset, entries in per_sample.items():
        for e in entries:
            rows.append({
                "asset": asset,
                "view_id": e["idx"],
                "cd": float(e["cd"]),
                "emd": float(e["emd"]),
                "f1": float(e["f1"]),
            })
            
    df = pd.DataFrame(rows)
    df["asset"] = pd.Categorical(df["asset"], 
                                 categories=sorted(df["asset"].unique()),
                                 ordered=True)
    
    return df

def save_boxplot(df, metric, out_path, title, ylabel):
    assert metric in df.columns, f"Invalid metric: {metric}"

    plot_df = df[np.isfinite(df[metric])].copy()

    plt.figure(figsize=(4.5, 5.5))
    ax = sns.boxplot(
        data=plot_df,
        x="asset",
        y=metric,
        showfliers=True,
        whis=1.5
    )

    ax.set_xlabel('Asset Class')
    ax.set_ylabel(ylabel)
    if title is None:
        title = f'{metric.upper()} distribution across viewpoints (per asset)'
    ax.set_title(title)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 12,
        "axes.titlesize": 16,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 12})
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()

def _compute_error_colormap(pred, gt, cmap='viridis', vmax=None):
    '''Return Nx3 colors for pred ponts = nearest distance to GT mapped to colormap'''
    pred_np =  np.asarray(pred.points)
    gt_np = np.asarray(gt.points)

    if pred_np.size == 0 or gt_np.size == 0:
        return np.zeros((pred_np.shape[0], 3), dtype=np.float32), np.zeros((pred_np.shape[0],), dtype=np.float32)
    
    tree = o3d.geometry.KDTreeFlann(pred) #o3d.geometry.KDTreeFlann(gt)

    dists = np.zeros((pred_np.shape[0],), dtype=np.float32)

    for i, p in enumerate(gt_np): #enumerate(pred_np):
        _, idx, dist2 = tree.search_knn_vector_3d(p, 1)
        dists[i] = np.sqrt(dist2[0]) if len(dist2) > 0 else 0.0

    if vmax is None:
        vmax = max(1e-6, np.percentile(dists, 95))

    norm = np.clip(dists / float(vmax), 0.0, 1.0)
    cmap_func = plt.get_cmap(cmap)
    colors = cmap_func(norm)[:, :3]
    return colors.astype(np.float64), dists

def _render_pcd_to_image(pcd,
                         center,
                         cam_pos,
                         cam_up, 
                         radius, 
                         distance,
                         width=512,
                         height=512,
                         point_size=3.5,
                         visible=False
                         ):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=visible)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    
    front = (center - cam_pos).astype(np.float64)
    norm = np.linalg.norm(front) + 1e-12
    front = front / norm

    ctr.set_lookat(center.tolist())
    ctr.set_front(front.tolist())
    ctr.set_up(cam_up.tolist())

    zoom = 0.85 * (radius / distance)
    ctr.set_zoom(float(zoom))

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = float(point_size)

    vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()

    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    return img8 

def _make_camera_params_from_gt(gt_pcd, cam_offset_factor=(0.8, -1.0, 1.6),
                                fov_deg=60.0,
                                width=512, height=512,
                                point_size=3.5,
                                visible=False):
    """
    Create a PinholeCameraParameters sampled from a GT visualizer. This returns
    a PinholeCameraParameters object that can be reused on other Visualizers so
    they use identical extrinsic+intrinsic.
    """
    # compute bbox anchor
    bbox = gt_pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = float(np.linalg.norm(extent) * 0.5)
    radius = radius if radius > 0 else 1.0

    cam_offset = np.array([cam_offset_factor[0] * radius,
                           cam_offset_factor[1] * radius,
                           cam_offset_factor[2] * radius], dtype=np.float64)
    

    z = 2*radius
    xy = np.array([1.0, 0.0], dtype=np.float64) * radius
    yaw_deg=45.0
    if abs(yaw_deg) > 1e-6:
        th = np.deg2rad(yaw_deg)
        Rz = np.array([[np.cos(th), -np.sin(th)],
                       [np.sin(th), np.cos(th)]], dtype=np.float64)
        xy = Rz.dot(xy)
    


    cam_pos = center + np.array([xy[0], xy[1], z], dtype=np.float64) #cam_offset
    cam_up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)

    # Create a short-lived visualizer with the GT cloud to set a camera
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=visible)
    vis.add_geometry(gt_pcd)

    ctr = vis.get_view_control()

    # compute normalized front vector (camera -> center)
    front = (center - cam_pos).astype(np.float64)
    front /= (np.linalg.norm(front) + 1e-12)

    # Set the camera explicitly
    ctr.set_lookat(center.tolist())
    ctr.set_front(front.tolist())
    ctr.set_up(cam_up.tolist())
    print(f"Camera set up to {cam_up.tolist()}")
    # optionally tune zoom a bit; this is still needed to get proper scale
    ctr.set_zoom(0.75)

    # convert to pinhole parameters (this captures the extrinsic matrix used)
    params = ctr.convert_to_pinhole_camera_parameters()

    # Replace intrinsics with a controlled intrinsic using a simple fov model
    # compute fx/fy from requested fov and image width
    fov_rad = np.deg2rad(float(fov_deg))
    fx = fy = 0.5 * width / np.tan(0.5 * fov_rad)
    cx = width * 0.5
    cy = height * 0.5
   
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    params.intrinsic = intr
    params.intrinsic.width = int(width)
    params.intrinsic.height = int(height)
    # clean up
    vis.destroy_window()

    # Return anchor info so caller can reuse center/radius if needed
    anchor = {"params": params, "center": center, "radius": radius, "cam_pos": cam_pos, "cam_up": cam_up}
    return anchor
def _render_with_camera_params(pcd, params, width=512, height=512, point_size=3.5, visible=False):
    """
    Render a single point cloud using the provided PinholeCameraParameters.
    """
    vis = o3d.visualization.Visualizer()
    win_w = int(getattr(params.intrinsic, "width", 0))
    win_h = int(getattr(params.intrinsic, "height", 0))
    vis.create_window(width=win_w, height=win_h, visible=visible)
    vis.add_geometry(pcd)

    print(f"[DEBUG] Render window size = ({win_w} {win_h}), params.intrinsic size = ({params.intrinsic.width} {params.intrinsic.height})")

    ctr = vis.get_view_control()
    # apply the previously-captured parameters exactly
    try:
        print("CONVERT...")
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    except Exception:
        # fallback: if convert_from fails for some Open3D installs, try set_lookat/front/up
        cam = params.extrinsic
        # still attempt approximate fallback
        # (we expect convert_from_pinhole_camera_parameters to work in most setups)
        pass

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = float(point_size)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def render_triplet_from_pcds(partial_pcd_path,
                             gt_pcd_path,
                             out_path,
                             predictor,
                             asset,
                             cd,
                             f1,
                             sel_type,
                             include_error=True,
                             panel_size=(512, 512),
                             point_size=3.0,
                             title_font_path=None
                             ):
    
    assert os.path.exists(partial_pcd_path), f"Partial PCD {partial_pcd_path} not found"
    assert os.path.exists(gt_pcd_path), f"GT PCD {gt_pcd_path} not found"

    # Load PCDs
    partial_pcd = o3d.io.read_point_cloud(partial_pcd_path)
    gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
    
    partial_pts = np.asarray(partial_pcd.points)
    gt_pts = np.asarray(gt_pcd.points)

    print("PARTIAL: n_pts", partial_pts.shape[0],
      "centroid", partial_pts.mean(axis=0) if partial_pts.size else None,
      "extent", (partial_pts.max(axis=0)-partial_pts.min(axis=0)) if partial_pts.size else None)
 
    print("GT:      n_pts", gt_pts.shape[0],
      "centroid", gt_pts.mean(axis=0) if gt_pts.size else None,
      "extent", (gt_pts.max(axis=0)-gt_pts.min(axis=0)) if gt_pts.size else None)
    
    # Compute prediction
    input = IO.get(partial_pcd_path).astype(np.float32)
    gt_norm = IO.get(gt_pcd_path).astype(np.float32)

    gt_norm, input, c = _norm_from_partial(gt_norm, input)
    complete = predictor.predict(input)

    # Denormalize
    complete = complete * 2.0 + c

    # Convert to open3d pcd
    complete_pcd = o3d.geometry.PointCloud()
    complete_pcd.points = o3d.utility.Vector3dVector(complete)
    complete_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (complete.shape[0], 1)))

    # # Compute camera anchor
    # gt_np = np.asarray(gt_pcd.points)
    # if gt_np.size == 0:
    #     center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    #     radius = 1.0
    # else:
    #     center = gt_np.mean(axis=0)
    #     radius = float(np.max(np.linalg.norm(gt_np - center, axis=1)))
    #     radius = radius if radius > 0 else 1.0
    
    # w, h = panel_size

    # Align partial and completion to gt centroid
    # partial_cent = partial_pts.mean(axis=0)
    # complete_cent = complete.mean(axis=0)
    # gt_cent = gt_pts.mean(axis=0)

    # partial_pcd = partial_pcd.translate(gt_cent - partial_cent)
    # complete_pcd = complete_pcd.translate(gt_cent - complete_cent)

    # bbox = gt_pcd.get_axis_aligned_bounding_box()
    # center = bbox.get_center()
    # extent = bbox.get_extent()
    # radius = float(np.linalg.norm(extent) * 0.5)
    # distance = 1.1 * radius 

    # cam_offset = np.array([0.8 * radius, -1.0 * radius, 1.6 * radius], dtype=np.float64)
    # cam_pos = center + cam_offset
    # cam_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    # img_partial = _render_pcd_to_image(partial_pcd, center, cam_pos, cam_up, radius, distance, width=w, height=h, point_size=point_size)
    # img_gt = _render_pcd_to_image(gt_pcd, center, cam_pos, cam_up, radius, distance, width=w, height=h, point_size=point_size)
    # img_complete = _render_pcd_to_image(complete_pcd, center, cam_pos, cam_up, radius, distance, width=w, height=h, point_size=point_size)

    # panels = [Image.fromarray(img_partial), Image.fromarray(img_complete), Image.fromarray(img_gt)]
    bbox = gt_pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = float(np.linalg.norm(extent) * 0.5)
    radius = radius if radius > 0 else 1.0
    distance = 1.5 * radius

    # pick a 3/4 top view (positive Z so you see the top)
    cam_offset = np.array([0.8 * radius, -1.0 * radius, 1.6 * radius], dtype=np.float64)
    cam_pos = center + cam_offset
    cam_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

    # Debug prints (optional)
    print("CAM anchor center:", center, "radius:", radius)
    print("cam_pos:", cam_pos, "cam_up:", cam_up)

    # Render helper that explicitly sets camera parameters (do not change the cloud)
    # def _render_pcd_with_fixed_camera(pcd, center, cam_pos, cam_up, width=512, height=512, point_size=3.5, visible=False):
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window(width=width, height=height, visible=visible)
    #     vis.add_geometry(pcd)
    #     ctr = vis.get_view_control()

    #     # compute normalized front vector from camera -> center
    #     front = (center - cam_pos).astype(np.float64)
    #     front /= (np.linalg.norm(front) + 1e-12)

    #     # Set camera explicitly — this makes the view deterministic across pcds
    #     try:
    #         ctr.set_lookat(center.tolist())
    #         ctr.set_front(front.tolist())
    #         ctr.set_up(cam_up.tolist())
    #     except Exception:
    #         # older Open3D versions sometimes behave differently — ignore and keep going
    #         pass

    #     # deterministic zoom; tweak 0.6-0.85 depending how tightly you want to frame
    #     ctr.set_zoom(0.75)

    #     opt = vis.get_render_option()
    #     opt.background_color = np.asarray([1.0, 1.0, 1.0])
    #     opt.point_size = float(point_size)

    #     vis.update_geometry(pcd)
    #     vis.poll_events()
    #     vis.update_renderer()
    #     img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    #     vis.destroy_window()
    #     return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # Use the function on the raw point clouds (no translation)
    w, h = panel_size
    anchor = _make_camera_params_from_gt(gt_pcd, cam_offset_factor=(0.8, -1.0, 1.6),
                                        fov_deg=60.0, width=w, height=h,
                                        point_size=point_size, visible=False)
    params = anchor['params']

    # 2) render each cloud using the exact same params (no translations)
    img_partial  = _render_with_camera_params(partial_pcd,  params, width=w, height=h, point_size=point_size, visible=False)
    img_complete = _render_with_camera_params(complete_pcd, params, width=w, height=h, point_size=point_size, visible=False)
    img_gt       = _render_with_camera_params(gt_pcd,       params, width=w, height=h, point_size=point_size, visible=False)

    # Now compose panels as you already do
    panels = [Image.fromarray(img_partial), Image.fromarray(img_complete), Image.fromarray(img_gt)]
    # img_partial  = _render_pcd_with_fixed_camera(partial_pcd,  center, cam_pos, cam_up, width=w, height=h, point_size=point_size)
    # img_complete = _render_pcd_with_fixed_camera(complete_pcd, center, cam_pos, cam_up, width=w, height=h, point_size=point_size)
    # img_gt       = _render_pcd_with_fixed_camera(gt_pcd,       center, cam_pos, cam_up, width=w, height=h, point_size=point_size)

    # Then combine panels exactly like you already do
    panels = [Image.fromarray(img_partial), Image.fromarray(img_complete), Image.fromarray(img_gt)]

    pred_error_stats = None
    if include_error: 
        err_colors, dists = _compute_error_colormap(complete_pcd, gt_pcd, cmap='viridis', vmax=None)
        
        complete_pts = np.asarray(complete_pcd.points)
        pred_err_pcd = o3d.geometry.PointCloud()
        pred_err_pcd.points = o3d.utility.Vector3dVector(complete_pts.astype(np.float64))
        pred_err_pcd.colors = o3d.utility.Vector3dVector(err_colors)
        img_err = _render_with_camera_params(pred_err_pcd, params, width=w, height=h, point_size=3.5, visible=False)
        panels.append(Image.fromarray(img_err))
        pred_error_stats = {"mean": dists.mean(), "max": float(np.max(dists)), "min": float(np.min(dists))}
    pad_between = 0
    grid_w = 2 * w + pad_between
    grid_h = 2 * h + pad_between
    #num_panels = len(panels)
    #total_w = w * num_panels

    try: 
        if title_font_path and os.path.exists(title_font_path):
            title_font = ImageFont.truetype(title_font_path,  20)
            caption_font  = ImageFont.truetype(title_font_path,  15)
        else:
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  20)
                caption_font  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  15)
            except Exception:
                title_font = ImageFont.load_default()
                caption_font = ImageFont.load_default()
    except Exception:
        title_font = ImageFont.load_default()
        caption_font = ImageFont.load_default()

    padding = 0

    title_height = int(5)
    caption_height = int(30)
    footer_padding = 30
    final_w = grid_w
    final_h = title_height + grid_h + caption_height + footer_padding
    out_image = Image.new("RGB", (final_w, final_h), (255, 255, 255))
    draw = ImageDraw.Draw(out_image)

    y_panels = title_height
    #out_image.paste(panels[0], (0, y_panels))
    #out_image.paste(panels[1], (w, y_panels))
    #out_image.paste(panels[3], (0, y_panels + h))
    #out_image.paste(panels[2], (w, y_panels + h))
   # for i, p in enumerate(panels):
    #    out_image.paste(p, (i * w, y_panels))

    # Title text
    cd_txt = f"CD: {cd:.2f} mm" if cd is not None else "CD: N/A"
    f1_txt = f"F1: {f1:.2f}" if f1 is not None else "F1: N/A"
    if sel_type == "lowest":
        sel_type = "Best"
    elif sel_type == "median":
        sel_type = "Median"
    elif sel_type == "highest":
        sel_type = "Worst"
    elif "outlier" in sel_type:
        sel_type = sel_type.capitalize()
    else:
        sel_type = "Unknown"

    title_txt = f"{asset.capitalize()} {sel_type.capitalize()} ({cd_txt}, {f1_txt})"
    tmp_img = Image.new("RGB", (10, 10))
    tmp_draw = ImageDraw.Draw(tmp_img)
    bbox_title = tmp_draw.textbbox((0, 0), title_txt, font=title_font)

    title_text_h = bbox_title[3] - bbox_title[1]
    title_pad_top = 4
    title_pad_bottom =  6
    title_height = title_text_h + title_pad_top + title_pad_bottom
    final_h = title_height + grid_h + caption_height + footer_padding
    out_image = Image.new("RGB", (final_w, final_h), (255, 255, 255))
    draw = ImageDraw.Draw(out_image)

    y_panels = title_height
    out_image.paste(panels[0], (0, y_panels))
    out_image.paste(panels[1], (w, y_panels))
    out_image.paste(panels[3], (0, y_panels + h))
    out_image.paste(panels[2], (w, y_panels + h))
    bbox = draw.textbbox((0, 0), title_txt, font=title_font)
    title_w = bbox[2] - bbox[0]
    title_h = bbox[3] - bbox[1]
    title_x = max(0, (final_w - title_w) // 2)
    title_y = title_pad_top #max(4, (title_height - title_h) // 2)
    draw.text((title_x, title_y), title_txt, fill=(0, 0, 0), font=title_font)

    caption_labels = ["Part.", "Pred."]
    if include_error:
        caption_labels.append(f"Err. (mean={2*pred_error_stats['mean']:.2f} mm, max={2 * pred_error_stats['max']:.2f} mm)")
    caption_labels.append("GT")
    #cap_y = y_panels + grid_h + (caption_height - 12) // 2
    for i, label in enumerate(caption_labels):
        row = i // 2
        col = i % 2

        cell_x = col * (w + pad_between)
        cell_y_top = y_panels + row * (h + pad_between)

        bbox = draw.textbbox((0, 0), label, font=caption_font)
        cap_w = bbox[2] - bbox[0]
        cap_h = bbox[3] - bbox[1]
        
        cap_x = cell_x + (w - cap_w) // 2 #i * w + (w - cap_w) // 2
        cap_y = cell_y_top + h + max(2, (caption_height - cap_h) // 2)
        draw.text((cap_x, cap_y), label, fill=(0, 0, 0), font=caption_font)


    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_image.save(out_path)

    info = {
        "center": center.tolist(),
        "radius": radius,
        "cam_pos": cam_pos.tolist(),
        "panels": len(panels),
        "pred_error": pred_error_stats
    }

    print(f"Render saved {out_path}")

    return out_path, info

def compute_samples(df, 
                    predictor,
                    out_csv="selected_samples.csv",
                    job_out_dir="metrics",
                    metric_col="cd"
                    ):
    # For CD:
    # Get lowest performing
    # Get median
    # Get highest performing sample (non-outlier)
    # Get all outliers
    assert metric_col in df.columns, f"Metric {metric_col} not found"
    picks = []

    grouped = df.groupby("asset")

    for asset, g in grouped:
        vals = g[metric_col].dropna().values
        if len(vals) == 0:
            continue

        q1 = np.percentile(vals, 25)
        q2 = np.percentile(vals, 50)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1

        outlier_high = q3 + 1.5 * iqr
        outlier_low = q1 - 1.5 * iqr

        # All outliers
        # Highest non-outlier
        non_outlier_mask = (g[metric_col] >= outlier_low) & (g[metric_col] <= outlier_high)
        outlier_mask = (g[metric_col] < outlier_low) | (g[metric_col] > outlier_high)

        non_outlier_df = g[non_outlier_mask].copy()
        outlier_df = g[outlier_mask].copy()

        if len(non_outlier_df) == 0:
            non_outlier_df = g.copy()

        highest_non_outlier = float(non_outlier_df[metric_col].max())
        lowest_non_outlier = float(non_outlier_df[metric_col].min())

        # Find the row
        def find_closest_row(value, kind):
            idx = (g[metric_col] - value).abs().idxmin()
            row = g.loc[idx]
            return idx, row
        
        try: 
            idx_min, row_min = find_closest_row(lowest_non_outlier, "lowest")
            picks.append((asset, metric_col, "lowest", idx_min, row_min))
        except: 
            print(f"[WARN] Couldn't pick lowest non-outlier for {asset}")

        try: 
            idx_med, row_med = find_closest_row(q2, "median")
            picks.append((asset, metric_col, "median", idx_med, row_med))
        except:
            print(f"[WARN] Couldn't pick median for {asset}")

        try: 
            idx_max, row_max = find_closest_row(highest_non_outlier, "highest")
            picks.append((asset, metric_col, "highest", idx_max, row_max))
        except:
            print(f"[WARN] Couldn't pick highest non-outlier for {asset}")

        # Outliers
        for oi, orow in outlier_df.iterrows():
            picks.append((asset, metric_col, "outlier", oi, orow))
        
    # Build dataframe
    pick_rows = []
    for asset, metric_used, sel_type, idx, row in picks:
        all_metrics = {c: float(row[c]) for c in df.columns if c in ("cd", "emd", "f1")}

        pick_row = {
            "asset": asset,
            "selection_type": sel_type,
            "selection_metric": metric_used,
            "df_index": int(idx),
            "view_id": int(row["view_id"]),
        }

        # Merge metric values
        pick_row.update(all_metrics)
        pick_rows.append(pick_row)

    picks_df = pd.DataFrame(pick_rows)

    # Save CSV
    picks_df.to_csv(out_csv, index=False)
    print(f"[RESULT] Saved {out_csv}")


    # Render and save as: asset_grid
    grouped = picks_df.groupby("asset")
    for asset, g in grouped:
        outlier_i = 1
        for _, row in g.iterrows():
            sel_type = row["selection_type"]
            cd = float(row["cd"])
            f1 = float(row["f1"])
            view_id = int(row["view_id"])

            partial_pcd_path = os.path.join("data", "NRG", "projected_partial_noise", asset, asset, "models", f"{view_id}.pcd")
            gt_pcd_path = os.path.join("data", "NRG", "NRG_pc", f"{asset}-{asset}-{view_id}.pcd")

            out_path = os.path.join(job_out_dir, f"{asset}_{sel_type}_{view_id}_grid.pdf")
            if sel_type=="outlier":
                sel_type = f"outlier {outlier_i}"
                outlier_i = outlier_i + 1

            render_triplet_from_pcds(partial_pcd_path,
                                 gt_pcd_path,
                                 out_path,
                                 predictor,
                                 asset,
                                 cd=cd,
                                 f1=f1,
                                 sel_type=sel_type,
                                 include_error=True,
                                 panel_size=(512, 512),
                                 point_size=2.0,)
    return picks_df


def main(cfg_path,
         ckpt_path,
         test_txt_path,
         out_path,
         ):

    assert os.path.exists(cfg_path), "Config file missing"
    assert os.path.exists(ckpt_path), "Checkpoint missing"


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # ---- build model ----
    config = cfg_from_yaml_file(cfg_path)
    model = builder.model_builder(config.model)
    builder.load_model(model, ckpt_path)
    model.to(device)
    model.eval()

    # # ---- predictor wrapper ----
    predictor = AdaPoinTrPredictor(model, normalize=False)

    # Loop through each taxonomy and model, and run inference, computing F1 and CD, and plotting distribution, and also saving 10th, 50th, and 90th percentiles
    with open(test_txt_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    file_list = []
    for line in lines:
        # taxonomy-model-00042
        line = line.strip()
        taxonomy_id = line.split('-')[0].split('/')[-1]
        model_id = line.split('-')[1]
        view_id = int(line.split('-')[2].split('.')[0])


        file_list.append({
            "taxonomy_id": taxonomy_id,
            "model_id": model_id,
            "view_id": view_id,
            "file_path": line,
        })

    print(f"[DATASET] {len(file_list)} samples were loaded")

    per_sample = {}

    for sample in file_list:
        print(f"Processing {sample['file_path']}...")
        # Open partial PCD and run inference
        # file_path: NRG_pc/glovebox-glovebox-215.pcd
        # partial_path: projected_partial_noise/glovebox/glovebox/models/215.pcd
        file_path = os.path.join("data", "NRG", sample['file_path'])
        partial_path = os.path.join("data", "NRG", "projected_partial_noise", sample['taxonomy_id'], sample['model_id'], "models", f"{sample['view_id']}.pcd")
   
        partial = IO.get(partial_path).astype(np.float32)
        gt0 = IO.get(file_path).astype(np.float32)

        print(f"partial {partial_path}, gt {file_path}")
        gt, partial_norm, c = _norm_from_partial(gt0, partial)
        
        # Predictor returns points normalized
        complete = predictor.predict(partial_norm)

        # Compute CD and F1 between complete and gt
        _metrics = Metrics.get(torch.from_numpy(complete).float().cuda().unsqueeze(0), torch.from_numpy(gt).float().cuda().unsqueeze(0), require_emd=True)
        
        if sample['model_id'] not in per_sample.keys():
            per_sample[sample['model_id']] = []

        per_sample[sample['model_id']].append({
             'idx': sample['view_id'], 
             'cd': 2 * _metrics[1],
             'emd': 2 * _metrics[3],
             'f1': _metrics[0],
             #'metrics': [float(x) for x in _metrics]
         })
        
    # Run analysis
    df = per_sample_to_df(per_sample)

    # Plot
    save_boxplot(df, "cd", os.path.join(out_path, "plots/boxplots/test_cd_boxplot.pdf"), title="Test Set CD by Asset", ylabel="CD (mm)")
    save_boxplot(df, "emd", os.path.join(out_path, "plots/boxplots/test_emd_boxplot.pdf"), title="Test Set EMD by Asset", ylabel="EMD (mm)")
    save_boxplot(df, "f1", os.path.join(out_path, "plots/boxplots/test_f1_boxplot.pdf"), title="Test Set F1 Score by Asset", ylabel="F1")

    # get iqs
    compute_samples(df, predictor,
                    os.path.join(out_path, "results_by_cd.csv"),
                    os.path.join(out_path, "plots/graphics"),
                    metric_col="cd")
    
    #compute_samples(df, predictor,
     #               os.path.join(out_path, "results_by_f1.csv"),
      #              os.path.join(out_path, "plots/graphics"),
       #             metric_col="f1")
    
   # compute_samples(df, predictor,
    #            os.path.join(out_path, "results_by_emd.csv"),
     #           os.path.join(out_path, "plots/graphics"),
      #          metric_col="emd")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path",
        required= True, 
        type = str, 
        help = 'Config file path')
    parser.add_argument(
        "--ckpt_path", 
        required = True,
        type = str, 
        help = 'Checkpoint file path')
    parser.add_argument(
        "--test_txt_path",
        required = True, 
        type = str, 
        help = 'Path to txt file containing list of test PCDs')
    parser.add_argument(
        "--out_path",
        type = str, 
        default="./results",
        help = 'Output path')
    args = parser.parse_args()

    main(cfg_path=args.cfg_path, ckpt_path=args.ckpt_path, test_txt_path=args.test_txt_path, out_path=args.out_path)
