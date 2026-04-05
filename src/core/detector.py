from ultralytics import YOLO
from src import config

class VehicleDetector:
    def __init__(self):
        # Load the YOLO model and fuse for faster inference
        self.model = YOLO(config.MODEL_NAME)
        self.model.fuse()
        self.classes = config.TARGET_CLASSES

    def detect(self, frame):
        # Run inference
        results = self.model(frame, classes=self.classes, verbose=False)[0]
        
        # Extract Tensors to NumPy
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        return boxes, confidences, class_ids
