import cv2
import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src import config
from src.core.detector import VehicleDetector
from src.core.tracker import VehicleTracker
from src.logic.homography import HomographyTransformer
from src.logic.speed_estimator import SpeedEstimator
from src.logic.direction import DirectionDetector
from src.logic.counter import TrafficCounter
from src.ui.visualizer import Visualizer

def main():
    # 1. Initialize Video
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {config.VIDEO_PATH}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.OUTPUT_PATH), exist_ok=True)
    out = cv2.VideoWriter(config.OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 2. Initialize Modules
    detector = VehicleDetector()
    tracker = VehicleTracker()
    homography = HomographyTransformer()
    speed_estimator = SpeedEstimator(fps)
    direction_detector = DirectionDetector()
    counter = TrafficCounter(config.COUNTING_LINE_Y)
    visualizer = Visualizer()

    print(f"🚀 Starting Production Traffic Monitor Pipeline...")
    print(f"Input: {config.VIDEO_PATH}")
    print(f"Output: {config.OUTPUT_PATH}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # A. Detect
        boxes, confidences, class_ids = detector.detect(frame)

        # B. Track
        tracked_detections = tracker.update(boxes, confidences, class_ids)

        # C. Process Physics per tracked object
        speeds = {}
        directions = {}
        counts = counter.counts 

        if tracked_detections.tracker_id is not None:
            for xyxy, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
                # We use the bottom-center of the bounding box to represent where the car touches the road
                x1, y1, x2, y2 = xyxy
                cx, cy = int((x1 + x2) / 2), int(y2)

                # 1. Homography Transform -> Get Real World Meters
                x_m, y_m = homography.transform_point(cx, cy)

                # 2. Estimate Speed
                speed = speed_estimator.update(track_id, x_m, y_m)
                speeds[track_id] = speed

                # 3. Detect Direction
                direction = direction_detector.update(track_id, cy)
                directions[track_id] = direction

                # 4. Count
                prev_y = direction_detector.last_y.get(track_id, cy)
                counts = counter.update(track_id, prev_y, cy, direction)

        # D. Visualize
        annotated_frame = visualizer.draw(frame, tracked_detections, speeds, directions, counts)

        # E. Output
        out.write(annotated_frame)
        # Headless mode for CLI
        # cv2.imshow("Traffic Monitor", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Processing Complete.")

if __name__ == "__main__":
    main()
