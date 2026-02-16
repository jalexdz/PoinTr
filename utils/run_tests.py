import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
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

def get_graphics():
    # Save partial | prediction | gt | heatmap
    pass

def compute_samples(df, 
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

    # # Create render jobs
    # job_paths = []
    # for _, r in picks_df.iterrows():
    #     sample_row = {
    #         "asset": r["asset"],
    #         "view_id": int(r["view_id"]),
    #         "selection_metric": r["selection_metric"],
    #         "selection_type": r["selection_type"],
    #         "cd": float(r["cd"]) if "cd" in r else None,
    #         "emd": float(r["emd"]) if "emd" in r else None,
    #         "f1": float(r["f1"]) if "f1" in r else None
    #     }

    #     job_path = get_graphics(sample_row, job_out_dir=job_out_dir,
    #                             note=f"selection_type={r['selection_type']}, metric={r['selection_metric']}")
        
    #     job_paths.append(job_path)

    # print(f"[RESULT] Created {len(job_paths)} jobs")

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
        gt, partial_norm = _norm_from_partial(gt0, partial)
        
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
    compute_samples(df,
                    os.path.join(out_path, "results_by_cd.csv"),
                    os.path.join(out_path, "plots/graphics"),
                    metric_col="cd")
    
    compute_samples(df,
                    os.path.join(out_path, "results_by_f1.csv"),
                    os.path.join(out_path, "plots/graphics"),
                    metric_col="f1")
    
    compute_samples(df,
                os.path.join(out_path, "results_by_emd.csv"),
                os.path.join(out_path, "plots/graphics"),
                metric_col="emd")

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
        required=True,
        type = str, 
        default="results",
        help = 'Output path')
    args = parser.parse_args()

    main(cfg_path=args.cfg_path, ckpt_path=args.ckpt_path, test_txt_path=args.test_txt_path, out_path=args.out_path)
