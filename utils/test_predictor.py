import os
import numpy as np
import torch
import open3d as o3d
from datasets.io import IO

from tools import builder
from utils.config import cfg_from_yaml_file
from tools.predictor import AdaPoinTrPredictor


def test_adapointr_inference_smoke():
    """
    Smoke test for AdaPoinTr inference:
    - model loads
    - checkpoint loads
    - inference runs
    - output shape is sane
    """

    # ---- paths (adjust if needed) ----
    cfg_path = "cfgs/ShapeNet55_models/AdaPoinTr.yaml"
    ckpt_path = "ckpts/AdaPoinTr_s55.pth"
    pcd_path = "demo/partial.pcd"

    assert os.path.exists(cfg_path), "Config file missing"
    assert os.path.exists(ckpt_path), "Checkpoint missing"
    assert os.path.exists(pcd_path), "Demo PCD missing"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- build model ----
    config = cfg_from_yaml_file(cfg_path)
    model = builder.model_builder(config.model)
    builder.load_model(model, ckpt_path)
    model.to(device)
    model.eval()

    # ---- predictor wrapper ----
    predictor = AdaPoinTrPredictor(model)

    # ---- load PCD ----
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    points = IO.get(pcd_path).astype(np.float32)

    assert points.ndim == 2
    assert points.shape[1] == 3
    assert points.shape[0] > 0

    # ---- run inference ----
    complete = predictor.predict(points)

    # ---- assertions ----
    assert isinstance(complete, np.ndarray)
    assert complete.ndim == 2
    assert complete.shape[1] == 3
    # assert complete.shape[0] > points.shape[0]  # completion expands

    # ---- optional artifact ----
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(complete)
    o3d.io.write_point_cloud("tests/output_complete.pcd", out_pcd)

    print(
        f"[OK] Input points: {points.shape[0]}, "
        f"Output points: {complete.shape[0]}"
    )


if __name__ == "__main__":
    test_adapointr_inference_smoke()
