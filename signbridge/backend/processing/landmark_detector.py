import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time

# Standard MediaPipe hand connections
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
])

class LandmarkDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # The model should be placed in backend/model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'hand_landmarker.task')
        if not os.path.exists(model_path):
            print(f"WARNING: MediaPipe task model not found at {model_path}.")
            self.detector = None
            return

        base_options = python.BaseOptions(model_asset_path=model_path)
        # We use RunningMode.IMAGE for static images (training) 
        # For video streaming, we need either VIDEO (with timestamps) or just stick to IMAGE.
        # Since static_image_mode=False is usually passed for video but IMAGE is simpler, 
        # let's map according to static_image_mode.
        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.static_image_mode = static_image_mode
        self.last_ts_ms = 0

    def detect(self, frame_rgb):
        if not self.detector:
            return [], None
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            if self.static_image_mode:
                results = self.detector.detect(mp_image)
            else:
                current_ts_ms = int(time.time() * 1000)
                if current_ts_ms <= self.last_ts_ms:
                    current_ts_ms = self.last_ts_ms + 1
                self.last_ts_ms = current_ts_ms
                results = self.detector.detect_for_video(mp_image, current_ts_ms)
        except Exception as e:
            print(f"Error during detection: {e}")
            return [], None
            
        landmarks = []
        if results and hasattr(results, 'hand_landmarks') and results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for lm in hand_landmarks:
                    landmarks.append([lm.x, lm.y, lm.z])
                # Break after first hand since max_num_hands=1
                break
                
        return landmarks, results

    def draw_landmarks(self, frame, results):
        if results and hasattr(results, 'hand_landmarks') and results.hand_landmarks:
            h, w, c = frame.shape
            for hand_landmarks in results.hand_landmarks:
                points = []
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))
                    
                # Draw connections
                for connection in HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if start_idx < len(points) and end_idx < len(points):
                        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
                        
                # Draw points
                for p in points:
                    cv2.circle(frame, p, 5, (0, 0, 255), -1)
        return frame
