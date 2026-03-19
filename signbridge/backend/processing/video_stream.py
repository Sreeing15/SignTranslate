import base64
import cv2
import numpy as np

class VideoStreamProcessor:
    @staticmethod
    def decode_base64_frame(data_url):
        """
        Convert base64 data url from frontend into an OpenCV format.
        """
        if ',' in data_url:
            encoded_data = data_url.split(',')[1]
        else:
            encoded_data = data_url
            
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Frontend canvas might give RGBA, convert BGR to RGB for MediaPipe
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        return None
