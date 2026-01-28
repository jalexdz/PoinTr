import argparse
import numpy as np
import torch
from tools import builder
from utils.config import * 

def stats(pc):
    """
    pc: (N,3) torch or numpy
    """
    pc = pc.detach().cpu().numpy() if torch.is_tensor(pc) else pc
    mu = pc.mean(axis=0)
    rmax = np.linalg.norm(pc - mu, axis=1).max()
    mn = pc.min(axis=0)
    mx = pc.max(axis=0)
    return mu, rmax, mn, mx

def main(args, config, max_batches=200):
    # Build dataset + dataloader (Projected_ShapeNet)
    (_, dataloader) = builder.dataset_builder(args, config.dataset.test)

    mus_p, mus_g = [], []
    rmax_p, rmax_g = [], []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        taxonomy_id, model_id, data = batch
        print("TESTING")
        # Projected_ShapeNet returns (partial, gt)
        partial, gt = data
        # shapes: [B, N, 3] — take first in batch
        partial = partial[0]
        gt = gt[0]

        mu_p, r_p, _, _ = stats(partial)
        mu_g, r_g, _, _ = stats(gt)

        mus_p.append(mu_p)
        mus_g.append(mu_g)
        rmax_p.append(r_p)
        rmax_g.append(r_g)

    mus_p = np.array(mus_p)
    mus_g = np.array(mus_g)
    rmax_p = np.array(rmax_p)
    rmax_g = np.array(rmax_g)

    print("\n=== Projected_ShapeNet stats ===")
    print("Samples:", len(mus_p))

    print("\nMean |centroid|")
    print("  partial:", np.mean(np.abs(mus_p), axis=0))
    print("  gt:     ", np.mean(np.abs(mus_g), axis=0))

    print("\nRmax (distance from centroid)")
    print("  partial: mean / min / max =",
          rmax_p.mean(), rmax_p.min(), rmax_p.max())
    print("  gt:      mean / min / max =",
          rmax_g.mean(), rmax_g.min(), rmax_g.max())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to yaml config (same one you use for test/train)")
    ap.add_argument("--max_batches", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--launcher", default="none")
    ap.add_argument("--local_rank", type=int, default=0)
    ap.add_argument("--use_gpu", action="store_true", default=True)
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--experiment_path", default=".")
    ap.add_argument("--distributed", action="store_true", default=False)
    args = ap.parse_args()

    # Load config yaml -> EasyDict (repo standard)
    config = get_config(args)

    main(args, config, max_batches=args.max_batches)
