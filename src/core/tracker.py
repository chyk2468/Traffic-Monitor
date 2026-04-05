import supervision as sv

class VehicleTracker:
    def __init__(self):
        # Using Supervision's highly optimized ByteTrack implementation
        self.tracker = sv.ByteTrack()

    def update(self, boxes, confidences, class_ids):
        if len(boxes) == 0:
            return sv.Detections.empty()

        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )
        
        # Update tracker and return tracked objects
        return self.tracker.update_with_detections(detections)
