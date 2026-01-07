import numpy as np

class AdaPoinTrPredictor:
    def __init__(self, model):
        self.model = model.eval()
        self.device = next(model.parameters()).device

    def predict(self,
                point_cloud: np.ndarray # Input partial point cloud
        ) -> np.ndarray:
        """
        Predict the complete point cloud from the partial input.

        Args:
            point_cloud: Input partial point cloud

        Returns:
            Predicted complete point cloud
        """

        print("Predicting...")

        complete_pc = None
        return complete_pc