from collections import deque
from collections import Counter

class SmoothingFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        
    def add_prediction(self, prediction):
        if prediction is not None:
            self.buffer.append(prediction)
            
    def get_smoothed_prediction(self):
        if not self.buffer:
            return None
            
        # Majority voting
        counter = Counter(self.buffer)
        most_common, count = counter.most_common(1)[0]
        
        # Require a minimum threshold (e.g., must be > 40% of the window)
        threshold = max(2, int(self.window_size * 0.4))
        if count >= threshold:
             return most_common
        return None
        
    def clear(self):
        self.buffer.clear()
