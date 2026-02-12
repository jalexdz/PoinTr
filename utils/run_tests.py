import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import torch
import open3d as o3d
from datasets.io import IO

from tools import builder
from utils.config import cfg_from_yaml_file
from tools.predictor import AdaPoinTrPredictor


def main(cfg_path,
         ckpt_path,
         test_txt_path,):
    # ---- paths (adjust if needed) ----

    # assert os.path.exists(cfg_path), "Config file missing"
    # assert os.path.exists(ckpt_path), "Checkpoint missing"

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # # ---- build model ----
    # config = cfg_from_yaml_file(cfg_path)
    # model = builder.model_builder(config.model)
    # builder.load_model(model, ckpt_path)
    # model.to(device)
    # model.eval()

    # # ---- predictor wrapper ----
    # predictor = AdaPoinTrPredictor(model)

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

        partial_path = os.path.join("data/NRG/projected_partial_noise", sample['taxonomy_id'], sample['model_id'], "models", f"{sample['view_id']}.pcd")
        # points = IO.get(partial_path).astype(np.float32)
        # gt = IO.get(sample['file_path']).astype(np.float32)
        
        #complete = predictor.predict(partial)

        # # Compute CD and F1 between complete and gt

        # cd = 0
        # f1 = 0

        # per_sample[sample['model_id']] = {
        #     'idx': sample['view_id'], 
        #     'CD': cd,
        #     'F1': f1,
        # }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path", 
        type = str, 
        help = 'Config file path')
    parser.add_argument(
        "--ckpt_path", 
        type = str, 
        help = 'Checkpoint file path')
    parser.add_argument(
        "--test_txt_path", 
        type = str, 
        help = 'Path to txt file containing list of test PCDs')
    args = parser.parse_args()

    main(cfg_path=args.cfg_path, ckpt_path=args.ckpt_path, test_txt_path=args.test_txt_path)