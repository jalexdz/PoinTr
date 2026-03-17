import argparse
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import open3d as o3d

from tools import builder
from utils.config import cfg_from_yaml_file
from tools.predictor import AdaPoinTrPredictor
from datasets.io import IO

# This computes zero-shot performance 
# Outputs just the visual results in a grid for every ablation/asset combination:
## | Part. | Pred. | 

def _norm_from_partial(partial: np.ndarray):
    centroid = np.mean(partial, axis=0)
    partial_norm = (partial - centroid) / 2.0
    return partial_norm.astype(np.float32), centroid

def _make_camera_params(pcd, fov_deg=60.0, width=512, height=512, visible=False):
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = float(np.linalg.norm(extent) * 0.5)
    radius = radius if radius > 0 else 1.0

    z = 2 * radius
    xy = np.array([1.0, 0.0], dtype=np.float64) * radius
    th = np.deg2rad(45.0)
    Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float64)
    xy = Rz.dot(xy)

    cam_pos = center + np.array([xy[0], xy[1], z], dtype=np.float64)
    cam_up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=visible)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    front = (center - cam_pos).astype(np.float64)
    front /= (np.linalg.norm(front) + 1e-12)
    ctr.set_lookat(center.tolist())
    ctr.set_front(front.tolist())
    ctr.set_up(cam_up.tolist())
    ctr.set_zoom(0.75)
    params = ctr.convert_to_pinhole_camera_parameters()
    fov_rad = np.deg2rad(float(fov_deg))
    fx = fy = 0.5 * width / np.tan(0.5 * fov_rad)
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, width * 0.5, height * 0.5)

    params.intrinsic = intr
    params.intrinsic.width = int(width)
    params.intrinsic.height = int(height)
    vis.destroy_window()

    return params

def _render(pcd,
            params,
            width=512,
            height=512,
            point_size=3.5,
            visible=False):
    win_w = int(getattr(params.intrinsic, "width", width))
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
    opt.point_size = float(point_size)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))

    vis.destroy_window()
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def _load_fonts():
    try:
        bold  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        reg   = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        bold = reg = small = ImageFont.load_default()
    return bold, reg, small

def render_grid(
        partial_pcd_path: str,
        predictors: list,
        out_path_root: str,
        out_path: str,
        asset: str,
        view_id: int,
        panel_w: int = 512,
        panel_h: int = 512,
        point_size: float = 2.0,
        row_label_w: int = 110,
        caption_h: int = 30,
        row_gap: int = 6,
        title_pad: int = 36,
        footer: int = 12
    ):
    assert os.path.exists(partial_pcd_path), \
        f"Partial PCD not found: {partial_pcd_path}"
    
    bold_font, reg_font, small_font = _load_fonts()

    n_rows = len(predictors)
    total_w = row_label_w + 2 * panel_w
    total_h = (title_pad 
               + n_rows * panel_h
               + (n_rows - 1) * row_gap
               + caption_h
               + footer)

    out_img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(out_img)

    title_txt = f'{asset.capitalize()} -- Example {view_id}'
    bb = draw.textbbox((0, 0), title_txt, font=bold_font)
    tx = max(0, (total_w - (bb[2] - bb[0])) // 2)
    draw.text((tx, 6), title_txt, fill=(0, 0, 0), font=bold_font)

    # Run predictions
    partial_pts = IO.get(partial_pcd_path).astype(np.float32)
    anchor_pcd = o3d.geometry.PointCloud()
    anchor_pcd.points = o3d.utility.Vector3dVector(partial_pts.astype(np.float64))

    cam_params = _make_camera_params(anchor_pcd, fov_deg=60.0, width=panel_w, height=panel_h)

    # Rows
    for row_i, (abl_name, predictor) in enumerate(predictors):
        y_top = title_pad + row_i * (panel_h + row_gap)

        # Inference
        partial_norm, centroid = _norm_from_partial(partial_pts)
        complete_pts = predictor.predict(partial_norm) * 2.0 + centroid

        part_pcd = o3d.geometry.PointCloud()
        part_pcd.points = o3d.utility.Vector3dVector(partial_pts.astype(np.float64))
        part_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.0, 0.0], (partial_pts.shape[0], 1))
        )

        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(complete_pts.astype(np.float64))
        pred_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 1.0, 0.0], (complete_pts.shape[0], 1))
        )

        # Render 
        img_part = _render(part_pcd,
                           cam_params,
                           width=panel_w,
                           height=panel_h,
                           point_size=point_size)
        
        img_pred = _render(pred_pcd,
                           cam_params,
                           width=panel_w,
                           height=panel_h,
                           point_size=point_size)
        
        out_img.paste(Image.fromarray(img_part), (row_label_w, y_top))
        out_img.paste(Image.fromarray(img_pred), (row_label_w + panel_w, y_top))

        lbb = draw.textbbox((0, 0), abl_name, font=small_font)
        lw = lbb[2] - lbb[0]
        lh = lbb[3] - lbb[1]
        lx = max(2, (row_label_w - lw) // 2)
        ly = y_top + (panel_h - lh) // 2
        draw.text((lx, ly), abl_name, fill=(0, 0, 0), font=small_font)

        if row_i < n_rows - 1:
            sep_y = y_top + panel_h + row_gap // 2
            draw.line([(row_label_w, sep_y), (total_w, sep_y)],
                      fill=(200, 200, 200), width=1)
            
    # Column captions
    cap_y0 = title_pad + n_rows * panel_h + (n_rows - 1) * row_gap

    for col_i, label in enumerate(["Part.", "Pred."]):
        cbb = draw.textbbox((0, 0), label, font=reg_font)
        cw = cbb[2] - cbb[0]
        ch = cbb[3] - cbb[1]
        cx = row_label_w + col_i * panel_w + (panel_w - cw) // 2
        cy = cap_y0 + max(2, (caption_h - ch) // 2)
        draw.text((cx, cy), label, fill=(0, 0, 0), font=reg_font)

    os.makedirs(os.path.dirname(out_path_root), exist_ok=True)
    out_img.save(out_path)
    print(f"  Saved {out_path}")
    return out_path

def main(ablation_configs: list[dict],
         assets: list[str],
         n_views: int,
         out_path: str,
         panel_size: int = 512,
         point_size: float = 2.0,
         txt_path: str = 'data/NRG_real/test.txt',
         data_dir: str = 'data/NRG_real/projected_partial_noise/',
         ):
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictors = []
    for abl in ablation_configs:
        print(f"Loading {abl['name']}...")

        config = cfg_from_yaml_file(abl['cfg_path'])
        model  = builder.model_builder(config.model)
        builder.load_model(model, abl['ckpt_path'])
        model.to(device)
        model.eval()

        predictor = AdaPoinTrPredictor(model, normalize=False)
        predictors.append((abl['name'], predictor))

    print(f"{len(predictors)} models loaded.")

    # Read test data
    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    file_list = []
    for line in lines:
        taxonomy_id = line.split('-')[0].split('/')[-1]
        model_id    = line.split('-')[1]
        view_id     = int(line.split('-')[2].split('.')[0])
        
        file_list.append({"taxonomy_id": taxonomy_id, "model_id": model_id,
                           "view_id": view_id, "file_path": line})

    print(f"  {len(file_list)} samples loaded")

    for file in file_list:
        out_file = os.path.join(
            out_path,
            file['taxonomy_id'],
            f"view_{file['view_id']:04d}.pdf"
        )

        try:
            render_grid(
                partial_pcd_path=os.path.join(data_dir, file['taxonomy_id'], file['model_id'], 'models', f"partial_{file['view_id']}.pcd"),
                predictors=predictors,
                out_path_root=out_path,
                out_path=out_file,
                asset=file['taxonomy_id'],
                view_id=file['view_id'],
                panel_w=panel_size,
                panel_h=panel_size,
                point_size=point_size,
            )
        except Exception as e:
            print(f"Error rendering {out_file}: {e}")

        print(f"Saved {out_file}")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot inference"
    )

    parser.add_argument(
        "--ablation", action="append", nargs=3,
        metavar=("NAME", "CFG", "CKPT"),
        help=(
            "One ablation: NAME cfg_path ckpt_path test_txt_path. Repeat for each ablation.\n"
            "  --ablation Baseline     cfgs/bl.yaml ckpts/bl.pth\n"
            "  --ablation 'Ablation 1' cfgs/a1.yaml ckpts/a1.pth\n"
            "  --ablation 'Ablation 2' cfgs/a2.yaml ckpts/a2.pth"
        ),
    )

    parser.add_argument(
        "--assets", nargs="+",
        default=["glovebox", "officechair", "woodentable", "trashcan"],
        help="Assets to test on"
    )
    parser.add_argument(
        "--n_views", type=int, default=5,
        help="Number of views to test on (set 0 to process all)"
    )

    parser.add_argument(
        "--out_path", type=str, default="./zeroshot_results",
        help="Root output directory"
    )

    parser.add_argument(
        "--panel_size", type=int, default=512,
        help="Panel size"
    )

    parser.add_argument(
        "--point_size", type=float, default=2.0,
        help="Rendrered point size"
    )

    parser.add_argument(
        "--txt_path", type=str, default=None,
        help="Path to txt file with test data"
    )
    args = parser.parse_args()

    if not args.ablation:
        parser.error("Provide at least one --ablation NAME CFG CKPT")
    if not args.txt_path:
        parser.error("Provide --txt_path")

    ablation_configs = [
        {"name": a[0], "cfg_path": a[1], "ckpt_path": a[2]}
        for a in args.ablation
    ]

    main(
        ablation_configs=ablation_configs,
        assets=args.assets,
        n_views=args.n_views if args.n_views > 0 else None,
        out_path=args.out_path,
        panel_size=args.panel_size,
        point_size=args.point_size,
        txt_path=args.txt_path
    )
