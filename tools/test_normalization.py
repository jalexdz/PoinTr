import numpy as np
import torch
from tools import builder

def stats(pc):
    pc = pc.detach().cpu().numpy() if torch.is_tensor(pc) else pc
    mu = pc.mean(axis=0)
    rmax = np.linalg.norm(pc - mu, axis=1).max()
    mn = pc.min(axis=0); mx = pc.max(axis=0)
    return mu, rmax, mn, mx

def main(args, config):
    (_, dl), = [builder.dataset_builder(args, config.dataset.test)]  # or val/train
    mus, rmaxs = [], []
    for i, batch in enumerate(dl):
        if i >= 200: break
        tax, mid, data = batch
        # adapt depending on dataset
        if isinstance(data, (list, tuple)):
            partial, gt = data
        else:
            partial, gt = data["partial"], data["gt"]

        mu_p, r_p, mn_p, mx_p = stats(partial[0])
        mu_g, r_g, mn_g, mx_g = stats(gt[0])

        mus.append((mu_p, mu_g))
        rmaxs.append((r_p, r_g))

    mus = np.array([[*a, *b] for a,b in mus], dtype=float)
    rmaxs = np.array(rmaxs, dtype=float)

    print("partial mean abs (avg):", np.mean(np.abs(mus[:,0:3]), axis=0))
    print("gt mean abs (avg):     ", np.mean(np.abs(mus[:,3:6]), axis=0))
    print("partial rmax: mean/min/max:", rmaxs[:,0].mean(), rmaxs[:,0].min(), rmaxs[:,0].max())
    print("gt rmax:      mean/min/max:", rmaxs[:,1].mean(), rmaxs[:,1].min(), rmaxs[:,1].max())
