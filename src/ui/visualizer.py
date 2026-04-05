import cv2
import numpy as np
from collections import deque
from src import config

class Visualizer:
    def __init__(self):
        self.trails = {} 

    def draw(self, frame, tracked_detections, speeds, directions, counts):
        # 1. Draw Homography ROI
        cv2.polylines(frame, [config.SOURCE_POLYGON.astype(np.int32)], True, (0, 255, 255), 2)
        
        # 2. Draw Counting Line
        h, w = frame.shape[:2]
        cv2.line(frame, (0, config.COUNTING_LINE_Y), (w, config.COUNTING_LINE_Y), (255, 0, 0), 2)

        # 3. Draw Traffic Dashboard
        cv2.rectangle(frame, (20, 20), (350, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Northbound: {counts['Northbound']}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Southbound: {counts['Southbound']}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 4. Draw Vehicles
        if tracked_detections.tracker_id is not None:
            for xyxy, track_id, class_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id, tracked_detections.class_id):
                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = int((x1 + x2) / 2), int(y2) # Bottom center for trails

                # Manage Trails
                if track_id not in self.trails:
                    self.trails[track_id] = deque(maxlen=config.TRAIL_LENGTH)
                self.trails[track_id].append((cx, cy))

                # Draw Trails
                points = list(self.trails[track_id])
                for i in range(1, len(points)):
                    thickness = int(np.sqrt(64 / float(i + 1)) * 2)
                    cv2.line(frame, points[i-1], points[i], (255, 200, 0), thickness)

                # Extract Data
                speed = speeds.get(track_id, 0)
                direction = directions.get(track_id, "Unknown")
                is_overspeeding = speed > config.SPEED_LIMIT_KMH

                # Color Logic
                color = (0, 0, 255) if is_overspeeding else (0, 255, 0)

                # Draw Box and Text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"#{track_id} {speed:.1f}km/h {direction}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame
