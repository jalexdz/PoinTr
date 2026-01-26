import numpy as np
import torch
from tools import builder

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
