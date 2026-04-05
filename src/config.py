import numpy as np
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

# Model & Video Paths
MODEL_NAME = os.path.join(DATA_DIR, "models", "yolov8s.pt")
VIDEO_PATH = os.path.join(DATA_DIR, "input", "input.mp4")
OUTPUT_PATH = os.path.join(DATA_DIR, "output", "output.mp4")

# 2 (Car), 3 (Motorcycle), 5 (Bus), 7 (Truck)
TARGET_CLASSES = [2, 3, 5, 7] 

# Physics & Tracking
SPEED_LIMIT_KMH = 80.0
TRAIL_LENGTH = 30 # Number of frames to keep in memory per vehicle

# ---------------------------------------------------------
# Homography (Bird's Eye View) Calibration
# ---------------------------------------------------------
# Format: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
SOURCE_POLYGON = np.array([
    [400, 300], [800, 300],  # Top of road segment
    [1000, 700], [200, 700]  # Bottom of road segment
], dtype=np.float32)

# Real-world measurements of that specific road segment in Meters
ROAD_WIDTH_M = 10.0   # e.g., 3 lanes approx 10 meters wide
ROAD_LENGTH_M = 40.0  # e.g., 40 meters long

TARGET_POLYGON = np.array([
    [0, 0], [ROAD_WIDTH_M, 0], 
    [ROAD_WIDTH_M, ROAD_LENGTH_M], [0, ROAD_LENGTH_M]
], dtype=np.float32)

# Counting Line (Y-coordinate line in the image)
COUNTING_LINE_Y = 500
