import cv2
import numpy as np
from src import config

class HomographyTransformer:
    def __init__(self):
        # Compute the perspective transform matrix once during initialization
        self.matrix = cv2.getPerspectiveTransform(config.SOURCE_POLYGON, config.TARGET_POLYGON)

    def transform_point(self, x, y):
        """Converts pixel coordinates to real-world meters."""
        pts = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.matrix)
        return transformed[0][0][0], transformed[0][0][1] # Returns (x_meter, y_meter)
