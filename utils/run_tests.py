#import seaborn as sns
#import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import torch
import open3d as o3d
from datasets.io import IO

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

    return gt0.astype(np.float32), partial0.astype(np.float32)

def main(cfg_path,
         ckpt_path,
         test_txt_path,):
    # ---- paths (adjust if needed) ----

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
        gt, partial_norm = _norm_from_partial(gt0, partial)
        
        # Predictor returns points normalized
        complete = predictor.predict(partial_norm)

        # Compute CD and F1 between complete and gt
        _metrics = Metrics.get(torch.from_numpy(complete).float().cuda().unsqueeze(0), torch.from_numpy(gt).float().cuda().unsqueeze(0), require_emd=True)
        
        if sample['model_id'] not in per_sample.keys():
            per_sample[sample['model_id']] = []

        per_sample[sample['model_id']].append({
             'idx': sample['view_id'], 
             'cd': 2*_metrics[1],
             'emd': _metrics[3],
             'f1': _metrics[0],
             #'metrics': [float(x) for x in _metrics]
         })
        
    # Run analysis
    for asset in per_sample.keys():
        for data in per_sample[asset]:
             print(data['idx'])

    print(per_sample.keys())
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
    args = parser.parse_args()

    main(cfg_path=args.cfg_path, ckpt_path=args.ckpt_path, test_txt_path=args.test_txt_path)
