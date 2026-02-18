import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np
import torch
import open3d as o3d
from open3d import camera as o3dcamera

from datasets.io import IO
from PIL import Image, ImageDraw, ImageFont
import copy

from tools import builder
from utils.config import cfg_from_yaml_file
from tools.predictor import AdaPoinTrPredictor
from extensions.chamfer_dist import ChamferDistanceL1
from utils.metrics import Metrics

def lookat_extrinsic(eye, center, up):
    eye = eye.astype(np.float64)
    center = center.astype(np.float64)
    up = up.astype(np.float64)

    z = eye - center
    z /= (np.linalg.norm(z) + 1e-12)

    x = np.cross(up, z)
    x /= (np.linalg.norm(x) + 1e-12)

    y = np.cross(z, x)

    R = np.vstack([x, y, z]).T
    t = -R.dot(eye)

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    return extrinsic

def intrinsics_from_fov(width, height, fov_deg):
    fov_rad = np.deg2rad(fov_deg)
    fy = 0.5 * height / np.tan(fov_rad / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    intr = o3dcamera.PinholeCameraIntrinsic(int(width), int(height), float(fx), float(fy), float(cx), float(cy))
    return intr
        
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
    
    gt_tree = o3d.geometry.KDTreeFlann(gt)
    dists = np.zeros((pred_np.shape[0],), dtype=np.float32)

    for i, p in enumerate(pred_np):
        _, idx, dist2 = gt_tree.search_knn_vector_3d(p, 1)
        dists[i] = np.sqrt(dist2[0]) if len(dist2) > 0 else 0.0

    if vmax is None:
        vmax = max(1e-6, np.percentile(dists, 95))

    norm = np.clip(dists / float(vmax), 0.0, 1.0)
    cmap_func = plt.get_cmap(cmap)
    colors = cmap_func(norm)[:, :3]
    return colors.astype(np.float64), dists

def _render_pcd_with_params(pcd,
                            center,
                            eye, 
                            up,
                            width=512,
                            height=512,
                            fov_deg=60,
                            point_size=3.5,
                            visible=False
                            ):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=visible)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()

    param = o3dcamera.PinholeCameraParameters()
    param.intrinsic = intrinsics_from_fov(width, height, fov_deg)
    param.extrinsic = lookat_extrinsic(np.asarray(eye), np.asarray(center), np.asarray(up))

    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = point_size

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

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
#     #ctr.set_lookat(cam_center.tolist())

#     #front = (cam_center - cam_pos)
#     #front = front / (np.linalg.norm(front) + 1e-12)

#     #ctr.set_front(front.tolist())
#     #ctr.set_up(cam_up.tolist())

#     ctr.set_lookat(center.tolist())
#     front = (center - cam_pos)
#     front /= np.linalg.norm(front)
#     ctr.set_front(front.tolist())
#     ctr.set_up(cam_up.tolist())
    
#     zoom = .75 #2.0 * (radius / distance)
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

    # Compute camera anchor
    gt_np = np.asarray(gt_pcd.points)
    if gt_np.size == 0:
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        radius = 1.0
    else:
        center = gt_np.mean(axis=0)
        radius = float(np.max(np.linalg.norm(gt_np - center, axis=1)))
        radius = radius if radius > 0 else 1.0

    #cam_offset = np.array([1.5 * radius, -1.5 * radius, 0.9 * radius], dtype=np.float64)

   # cam_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    w, h = panel_size

    def dump_stats(name, p):
        pts = np.asarray(p.points)
        print(f"{name} count={len(pts)} min={np.min(pts)} max={np.max(pts)}")

    dump_stats("partial", partial_pcd)
    dump_stats("gt", gt_pcd)
    dump_stats("complete", complete_pcd)

    bbox = gt_pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()

    bbox_margin = 0.05
    extent = extent * (1.0 + bbox_margin)
    radius = float(np.linalg.norm(extent) * 0.5)

    eye = center + np.array([1.5 * radius, -1.5 * radius, 0.9 * radius], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    fov_deg = 45.0
#    # distance = 1.1 * radius 
#     cam_offset = np.array([1.5 * radius, -1.5 * radius, 0.9 * radius], dtype=np.float64)
#     cam_pos = center + cam_offset
#     distance = float(np.linalg.norm(cam_offset))    
# # cam_dir = np.array([1.0, -1.0, 0.6])
#     # cam_dir = center + cam_dir * distance

#     cam_up = np.asarray([0.0, 0.0, 1.0])
    img_partial = _render_pcd_with_params(partial_pcd, center, eye, up, width=w, height=h, fov_deg=fov_deg, point_size=point_size, visible=False)
    img_complete = _render_pcd_with_params(complete_pcd, center, eye, up, width=w, height=h, fov_deg=fov_deg, point_size=point_size, visible=False)
    img_gt = _render_pcd_with_params(gt_pcd, center, eye, up, width=w, height=h, fov_deg=fov_deg, point_size=point_size, visible=False)
    # img_partial = _render_pcd_to_image(partial_pcd, center, cam_pos, cam_up, radius, distance, width=w, height=h, point_size=point_size)
    # img_gt = _render_pcd_to_image(gt_pcd, center, cam_pos, cam_up, radius, distance, width=w, height=h, point_size=point_size)
    # img_complete = _render_pcd_to_image(complete_pcd, center, cam_pos, cam_up, radius, distance, width=w, height=h, point_size=point_size)

    panels = [Image.fromarray(img_partial), Image.fromarray(img_complete), Image.fromarray(img_gt)]

    pred_error_stats = None
    if include_error: 
        err_colors, dists = _compute_error_colormap(complete_pcd, gt_pcd, cmap='viridis', vmax=None)
        
        complete_pts = np.asarray(complete_pcd.points)
        pred_err_pcd = o3d.geometry.PointCloud()
        pred_err_pcd.points = o3d.utility.Vector3dVector(complete_pts.astype(np.float64))
        pred_err_pcd.colors = o3d.utility.Vector3dVector(err_colors)
        #img_err = _render_pcd_to_image(pred_err_pcd, center, cam_pos, cam_up, radius, distance, width=w, height=h, point_size=3.5)
        img_err = _render_pcd_with_params(pred_err_pcd, center, eye, up, width=w, height=h, fov_deg=fov_deg, point_size=3.5, visible=False)
        panels.append(Image.fromarray(img_err))
        pred_error_stats = {"mean": dists.mean(), "max": float(np.max(dists)), "min": float(np.min(dists))}

    num_panels = len(panels)
    total_w = w * num_panels

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
    title_height = int(40)
    caption_height = int(28)
    footer_padding = 6

    final_h = title_height + h + caption_height + footer_padding
    out_image = Image.new("RGB", (total_w, final_h), (255, 255, 255))
    draw = ImageDraw.Draw(out_image)

    y_panels = title_height
    for i, p in enumerate(panels):
        out_image.paste(p, (i * w, y_panels))

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
    bbox = draw.textbbox((0, 0), title_txt, font=title_font)
    title_w = bbox[2] - bbox[0]
    title_h = bbox[3] - bbox[1]
    title_x = max(0, (total_w - title_w) // 2)
    title_y = max(4, (title_height - title_h) // 2)
    draw.text((title_x, title_y), title_txt, fill=(0, 0, 0), font=title_font)

    caption_labels = ["Part.", "Pred.", "GT"]
    if include_error:
        caption_labels.append(f"Err. (mean={2*pred_error_stats['mean']:.2f} mm, max={2 * pred_error_stats['max']:.2f} mm)")

    for i, label in enumerate(caption_labels):
        bbox = draw.textbbox((0, 0), label, font=caption_font)
        cap_w = bbox[2] - bbox[0]
        cap_h = bbox[3] - bbox[1]
        
        cap_x = i * w + (w - cap_w) // 2
        cap_y = y_panels + h + (caption_height - cap_h) // 2
        draw.text((cap_x, cap_y), label, fill=(0, 0, 0), font=caption_font)


    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_image.save(out_path)

    info = {
        "center": center.tolist(),
        "radius": radius,
        #"cam_pos": cam_pos.tolist(),
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
