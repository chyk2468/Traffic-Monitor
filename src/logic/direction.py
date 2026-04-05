class DirectionDetector:
    def __init__(self):
        self.directions = {} # track_id -> string
        self.last_y = {}     # track_id -> last Y pixel

    def update(self, track_id, current_y):
        prev_y = self.last_y.get(track_id)
        self.last_y[track_id] = current_y

        if prev_y is None:
            return self.directions.get(track_id, "Unknown")

        dy = current_y - prev_y
        
        # Deadzone to prevent jittering directions when stopped
        if abs(dy) < 0.5:
            return self.directions.get(track_id, "Unknown")

        # Down the screen = Southbound, Up the screen = Northbound
        direction = "Southbound" if dy > 0 else "Northbound"
        self.directions[track_id] = direction
        
        return direction
