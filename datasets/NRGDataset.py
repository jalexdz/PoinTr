import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *

# def fps_np(points: np.ndarray, n: int, rng: np.random.Generator):
#     """Simple numpy FPS. points: (N,3). Returns (n,3)."""
#     N = points.shape[0]
#     if N == 0:
#         return np.zeros((n, 3), dtype=np.float32)
#     if N <= n:
#         idx = rng.choice(N, n, replace=True)
#         return points[idx]

#     sel = np.empty((n,), dtype=np.int64)
#     sel[0] = rng.integers(0, N)
#     d2 = np.sum((points - points[sel[0]])**2, axis=1)
#     for i in range(1, n):
#         sel[i] = int(np.argmax(d2))
#         d2 = np.minimum(d2, np.sum((points - points[sel[i]])**2, axis=1))
#     return points[sel]

@DATASETS.register_module()
class NRG(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_root = config.COMPLETE_POINTS_ROOT
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.n_renderings = config.N_RENDERINGS if self.subset == 'train' else 1
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f"[DATASET] Open file {self.data_list_file}")
        with open(self.data_list_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        self.file_list = []
        for line in lines:
            # taxonomy-model-00042
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1]
            view_id = int(line.split('-')[2].split('.')[0])
    

            self.file_list.append({
                "taxonomy_id": taxonomy_id,
                "model_id": model_id,
                "view_id": view_id,
            })

        print(f"[DATASET] {len(self.file_list)} samples were loaded")

    def _get_transforms(self, subset):
            if subset == 'train':
                return data_transforms.Compose([{
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': 2048
                    },
                    'objects': ['partial']
                }, {
                    'callback': 'RandomMirrorPoints',
                    'objects': ['partial', 'gt']
                },{
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }])
            else:
                return data_transforms.Compose([{
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': 2048
                    },
                    'objects': ['partial']
                }, {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }])

    def _norm_from_partial(self, gt: np.ndarray, partial: np.ndarray):
        # Normalize from PARTIAL to GT
        centroid = np.mean(partial, axis=0)
        p0 = partial - centroid
        scale = np.max(np.linalg.norm(p0, axis=1)) + 1e-12

        partial0 = p0 / scale
        gt0 = (gt - centroid) / scale

        return gt0.astype(np.float32), partial0.astype(np.float32)

    # def __getitem__(self, idx):
    #     s = self.file_list[idx]
    #     taxonomy_id, model_id, view_id = s["taxonomy_id"], s["model_id"], s["view_id"]

    #     partial_rel = self.partial_fmt % (taxonomy_id, model_id, view_id)
    #     gt_rel = self.gt_fmt % (taxonomy_id, model_id)

    #     partial_path = os.path.join(self.data_root, partial_rel)
    #     gt_path = os.path.join(self.data_root, gt_rel)

    #     partial = IO.get(partial_path).astype(np.float32)
    #     gt = IO.get(gt_path).astype(np.float32)
  
    #     # joint normalization (gt-derived)
    #     gt, partial = self._norm_from_partial(gt, partial)

    #     # enforce fixed size
    #     partial = self._sample_to_n(partial)
    #     gt = self._sample_to_n(gt)


    #     partial = torch.from_numpy(partial).float()
    #     gt = torch.from_numpy(gt).float()
    #     data = {
    #                 'partial': partial,
    #                 'gt': gt
    #             }

    #     return taxonomy_id, model_id, data

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        #rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        gt_path = os.path.join(self.complete_points_root, 'NRG_pc',  sample['taxonomy_id'] + '-', sample['model_id'] + '.pcd')
        gt = IO.get(gt_path).astype(np.float32)

        partial_path = self.partial_points_path % (sample['taxonomy_id'], sample['model_id'], sample['view_id'])
        partial = IO.get(partial_path).astype(np.float32)


        if self.transforms is not None:
            data = self.transforms(data)

        gt, partial = self._norm_from_partial(gt, partial)
        
        data['gt'] = gt
        data['partial'] = partial
        assert data['gt'].shape[0] == self.npoints

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)
