from collections import deque
import numpy as np
from src import config

class SpeedEstimator:
    def __init__(self, fps):
        self.fps = fps
        self.history = {} # track_id -> deque of (x_m, y_m)
        self.speeds = {}  # track_id -> current speed

    def update(self, track_id, current_x_m, current_y_m):
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=config.TRAIL_LENGTH)

        self.history[track_id].append((current_x_m, current_y_m))

        # Need at least 5 frames to calculate a stable speed
        if len(self.history[track_id]) >= 5:
            p1 = np.array(self.history[track_id][0])
            p2 = np.array(self.history[track_id][-1])
            
            distance_m = np.linalg.norm(p2 - p1)
            time_s = len(self.history[track_id]) / self.fps
            
            speed_ms = distance_m / time_s
            speed_kmh = speed_ms * 3.6
            self.speeds[track_id] = speed_kmh
        else:
            self.speeds[track_id] = 0.0

        return self.speeds[track_id]
