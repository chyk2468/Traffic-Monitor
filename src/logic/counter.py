class TrafficCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.counts = {"Northbound": 0, "Southbound": 0}
        self.type_counts = {}
        self.crossed_ids = set()

    def update(self, track_id, prev_y, current_y, direction, vehicle_type):
        if track_id in self.crossed_ids or prev_y is None or direction == "Unknown":
            return self.counts, self.type_counts

        # Check if the bounding box bottom crossed the virtual Y-axis line
        if (prev_y < self.line_y and current_y >= self.line_y) or \
           (prev_y > self.line_y and current_y <= self.line_y):
            self.counts[direction] += 1
            self.type_counts[vehicle_type] = self.type_counts.get(vehicle_type, 0) + 1
            self.crossed_ids.add(track_id)

        return self.counts, self.type_counts
