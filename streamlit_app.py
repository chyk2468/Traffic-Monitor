import streamlit as st
import cv2
import tempfile
import os
import sys
import numpy as np

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

# --- Page Config ---
st.set_page_config(page_title="AI Traffic Monitor", page_icon="🚗", layout="wide")

st.title("🚀 AI-Powered Traffic Monitor Pipeline")
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("🔧 Configuration")
speed_limit = st.sidebar.slider("Speed Limit (km/h)", 20, 120, int(config.SPEED_LIMIT_KMH))
config.SPEED_LIMIT_KMH = speed_limit

uploaded_file = st.sidebar.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

# --- Main Dashboard Setup ---
col_video, col_stats = st.columns([3, 1])

with col_video:
    st.subheader("Live Processing Feed")
    video_placeholder = st.empty()

with col_stats:
    st.subheader("Traffic Statistics")
    north_stat = st.empty()
    south_stat = st.empty()
    type_stat_container = st.empty()

# --- Processing Pipeline ---
if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize Modules
    detector = VehicleDetector()
    tracker = VehicleTracker()
    homography = HomographyTransformer()
    speed_estimator = SpeedEstimator(fps)
    direction_detector = DirectionDetector()
    counter = TrafficCounter(config.COUNTING_LINE_Y)
    visualizer = Visualizer()

    st.success("Video Loaded. Processing...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # A. Detect & Track
        boxes, confidences, class_ids = detector.detect(frame)
        tracked_detections = tracker.update(boxes, confidences, class_ids)

        # B. Process Logic
        speeds = {}
        directions = {}
        counts = counter.counts 
        type_counts = counter.type_counts

        if tracked_detections.tracker_id is not None:
            for xyxy, track_id, class_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id, tracked_detections.class_id):
                x1, y1, x2, y2 = xyxy
                cx, cy = int((x1 + x2) / 2), int(y2)

                x_m, y_m = homography.transform_point(cx, cy)
                speed = speed_estimator.update(track_id, x_m, y_m)
                speeds[track_id] = speed
                direction = direction_detector.update(track_id, cy)
                directions[track_id] = direction

                prev_y = direction_detector.last_y.get(track_id, cy)
                vehicle_type = config.CLASS_NAMES.get(class_id, "Unknown")
                counts, type_counts = counter.update(track_id, prev_y, cy, direction, vehicle_type)

        # C. Visualize
        annotated_frame = visualizer.draw(frame, tracked_detections, speeds, directions, counts, type_counts)
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # D. Update Stats (Sidebar or Column)
        north_stat.metric("⬆️ Northbound", counts['Northbound'])
        south_stat.metric("⬇️ Southbound", counts['Southbound'])
        
        with type_stat_container.container():
            st.write("📋 **Counts by Type:**")
            for v_type, count in type_counts.items():
                st.write(f"- {v_type}: {count}")

    cap.release()
    tfile.close()
    os.unlink(tfile.name)
    st.success("✅ Video Processing Complete.")
else:
    st.info("👈 Please upload a video file to start monitoring.")
