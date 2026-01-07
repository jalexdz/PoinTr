import numpy as np
import torch

class AdaPoinTrPredictor:
    def __init__(self, model: torch.nn.Module,
                       n_points: int = 2048,
                       normalize: bool = True
        ) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.n_points = n_points
        self.normalize = normalize

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
        assert point_cloud.ndim == 2 and point_cloud.shape[1] == 3, "Input point cloud must be of shape (N, 3)"

        with torch.no_grad():
            xyz = torch.from_numpy(point_cloud).float().unsqueeze(0).to(self.device) # (1, N, 3)

            coarse, fine = self.model(xyz) # fine: (1, M, 3)
            complete_pc = fine.squeeze(0).cpu().numpy() # (M, 3)

        return complete_pc