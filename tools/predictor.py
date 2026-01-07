import numpy as np
import torch
from datasets.data_transforms import Compose

class AdaPoinTrPredictor:
    def __init__(self, model: torch.nn.Module,
                       n_points: int = 2048,
                       normalize: bool = False
        ) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.n_points = n_points
        self.normalize = normalize

        self.transform = Compose([{
            'callback': 'UpSamplePoints',
            'parameters': {
                'n_points': self.n_points
            },
            'objects': ['input']
        }, {
            'callback': 'ToTensor',
            'objects': ['input']
        }])

    def predict(self,
                point_cloud: np.ndarray # Input partial point cloud
        ) -> np.ndarray:
        """
        Predict the complete point cloud from the partial input.

        Args:
            point_cloud: (N, 3) input partial point cloud

        Returns:
            complete_pc: (M, 3) predicted complete point cloud
        """
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        assert point_cloud.ndim == 2 and point_cloud.shape[1] == 3, "Input point cloud must be of shape (N, 3)"

        if self.normalize:
            centroid = np.mean(point_cloud, axis=0)
            point_cloud = point_cloud - centroid
            m = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
            point_cloud = point_cloud / (m + 1e-8)

        pc_ndarray_normalized = self.transform({'input': point_cloud})

        with torch.no_grad():
            coarse, fine = self.model(pc_ndarray_normalized['input'].unsqueeze(0).to(self.device)) # fine: (1, M, 3)
            complete_pc = fine.squeeze(0).cpu().numpy() # (M, 3)

        if self.normalize:
            complete_pc = complete_pc * (m + 1e-8)
            complete_pc = complete_pc + centroid

        return complete_pc
