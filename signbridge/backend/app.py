import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import our processing and model classes
from processing.landmark_detector import LandmarkDetector
from processing.feature_extractor import FeatureExtractor
from processing.video_stream import VideoStreamProcessor
from model.predictor import HandGesturePredictor
from utils.smoothing import SmoothingFilter

app = Flask(__name__)
# Enable CORS for frontend to backend communication
CORS(app)

# Global instances
detector = LandmarkDetector(static_image_mode=True)
extractor = FeatureExtractor()

# Paths to trained models
gnn_model_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'gnn_model.pth')
xgb_model_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'xgb_model.json')

predictor = HandGesturePredictor(gnn_path=gnn_model_path, xgb_path=xgb_model_path)
smoothing_filter = SmoothingFilter(window_size=5)
is_running = False

@app.route('/start', methods=['POST'])
def start_recognition():
    global is_running
    is_running = True
    smoothing_filter.clear()
    return jsonify({"status": "started", "message": "Backend ready to process frames."}), 200

@app.route('/stop', methods=['POST'])
def stop_recognition():
    global is_running
    is_running = False
    smoothing_filter.clear()
    return jsonify({"status": "stopped", "message": "Backend stopped."}), 200

@app.route('/predict', methods=['POST'])
def predict_gesture():
    global is_running
    if not is_running:
        return jsonify({"error": "Service not started"}), 400
        
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
        
    # Decode base64 image
    base64_img = data['image']
    frame_rgb = VideoStreamProcessor.decode_base64_frame(base64_img)
    
    if frame_rgb is None:
         return jsonify({"error": "Failed to decode image"}), 400
         
    # 1. Detect Landmarks
    landmarks, _ = detector.detect(frame_rgb)
    
    if not landmarks:
        return jsonify({"prediction": None, "message": "No hand detected"}), 200
        
    # 2. Extract Features
    features = extractor.extract_features(landmarks)
    edges = extractor.get_edge_index()
    
    # 3. Predict
    prediction = predictor.predict(features, edges)
    
    # 4. Smooth prediction
    smoothing_filter.add_prediction(prediction)
    smoothed_pred = smoothing_filter.get_smoothed_prediction()
    
    return jsonify({
        "prediction": smoothed_pred, 
        "raw_prediction": prediction,
        "landmarks": landmarks # Return landmarks to the frontend
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
