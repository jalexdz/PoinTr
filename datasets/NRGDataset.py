import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS

def fps_np(points: np.ndarray, n: int, rng: np.random.Generator):
    """Simple numpy FPS. points: (N,3). Returns (n,3)."""
    N = points.shape[0]
    if N == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if N <= n:
        idx = rng.choice(N, n, replace=True)
        return points[idx]

    sel = np.empty((n,), dtype=np.int64)
    sel[0] = rng.integers(0, N)
    d2 = np.sum((points - points[sel[0]])**2, axis=1)
    for i in range(1, n):
        sel[i] = int(np.argmax(d2))
        d2 = np.minimum(d2, np.sum((points - points[sel[i]])**2, axis=1))
    return points[sel]

@DATASETS.register_module()
class NRG(data.Dataset):
    """
    Expects list lines like:  taxonomy-model-00042
    Uses config.PARTIAL_POINTS_PATH and config.COMPLETE_POINTS_PATH format strings:
      partial: <DATA_PATH> / (PARTIAL_POINTS_PATH % (taxonomy_id, model_id, view_id))
      gt:      <DATA_PATH> / (COMPLETE_POINTS_PATH % (taxonomy_id, model_id))
    """
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.partial_fmt = config.PARTIAL_POINTS_PATH
        self.gt_fmt = config.COMPLETE_POINTS_PATH

        self.subset = config.subset
        self.npoints = int(config.N_POINTS)

        self.data_list_file = os.path.join(self.data_root, f"{self.subset}.txt")

        self.use_fps = bool(getattr(config, "USE_FPS", True))
        self.seed = int(getattr(config, "SEED", 0))
        self.rng = np.random.default_rng(self.seed)

        print(f"[DATASET] Open file {self.data_list_file}")
        with open(self.data_list_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        self.file_list = []
        for line in lines:
            # taxonomy-model-00042
            parts = line.split("-")
            if len(parts) < 3:
                raise ValueError(
                    f"Bad line '{line}'. Expected 'taxonomy-model-00042'."
                )
            taxonomy_id = parts[0]
            model_id = parts[1]
            view_str = parts[2]
            fname = os.path.basename(view_str)
            view_id = int(fname.split("_")[0])  # handles 00042

            self.file_list.append({
                "taxonomy_id": taxonomy_id,
                "model_id": model_id,
                "view_id": view_id,
                "line": line,
            })

        print(f"[DATASET] {len(self.file_list)} samples were loaded")

    def _sample_to_n(self, pc: np.ndarray) -> np.ndarray:
        if pc.shape[0] == self.npoints:
            return pc.astype(np.float32)

        if self.use_fps:
            return fps_np(pc, self.npoints, self.rng).astype(np.float32)

        # random sample / pad
        N = pc.shape[0]
        if N >= self.npoints:
            idx = self.rng.choice(N, self.npoints, replace=False)
        else:
            idx = self.rng.choice(N, self.npoints, replace=True)
        return pc[idx].astype(np.float32)

    def _norm_like_shapenet_using_gt(self, gt: np.ndarray, partial: np.ndarray):
        centroid = np.mean(gt, axis=0)
        gt0 = gt - centroid
        m = np.max(np.sqrt(np.sum(gt0**2, axis=1)))
        scale = (m + 1e-12)
        gt0 = gt0 / scale
        partial0 = (partial - centroid) / scale
        return gt0.astype(np.float32), partial0.astype(np.float32)

    def __getitem__(self, idx):
        s = self.file_list[idx]
        taxonomy_id, model_id, view_id = s["taxonomy_id"], s["model_id"], s["view_id"]

        partial_rel = self.partial_fmt % (taxonomy_id, model_id, view_id)
        gt_rel = self.gt_fmt % (taxonomy_id, model_id)

        partial_path = os.path.join(self.data_root, partial_rel)
        gt_path = os.path.join(self.data_root, gt_rel)

        partial = IO.get(partial_path).astype(np.float32)
        gt = IO.get(gt_path).astype(np.float32)

        # enforce fixed size
        partial = self._sample_to_n(partial)
        gt = self._sample_to_n(gt)

        # joint normalization (gt-derived)
        gt, partial = self._norm_like_shapenet_using_gt(gt, partial)

        partial = torch.from_numpy(partial).float()
        gt = torch.from_numpy(gt).float()

        return taxonomy_id, model_id, partial, gt

    def __len__(self):
        return len(self.file_list)
