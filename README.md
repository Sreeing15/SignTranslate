# SignTranslate: Real-time AI Sign Language Translator

![SignTranslate](https://img.shields.io/badge/Status-Active-success)
![Python Backend](https://img.shields.io/badge/Backend-Flask-blue)
![Machine Learning](https://img.shields.io/badge/ML-PyTorch%20%7C%20XGBoost-orange)

**SignTranslate** is a real-time web application that translates sign language gestures into English text using your webcam. The application features a lightweight web frontend and a robust Python backend that processes video frames using Google's MediaPipe, structural modeling with Graph Neural Networks (GNN), and classification with XGBoost to deliver low-latency (<200ms) predictions.

## 🚀 Features

- **Real-Time Translation**: Detects hand gestures and converts them into text with high accuracy.
- **Advanced Architecture**: Combines MediaPipe for landmark extraction, a Graph Neural Network (GNN) for spatial feature processing, and XGBoost for robust classification.
- **Extensible & Modular**: Designed to be easily expanded for continuous gesture recognition (e.g., full sentences) and additional features like blink detection.
- **Latency Optimized**: Processing pipeline designed to return predictions in under 200ms.

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (WebRTC & Canvas API)
- **Backend / API**: Python 3, Flask, Flask-CORS
- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning Architecture**:
  - **PyTorch (GNN)**: Captures complex spatial dependencies between hand landmarks.
  - **XGBoost**: Highly accurate gradient-boosted tree model for the final gesture classification.

## 📁 Project Structure

```text
sign-translate/
├── signbridge/
│   ├── backend/
│   │   ├── app.py                     # Main Flask Application
│   │   ├── processing/                # MediaPipe Landmark & Feature Extraction
│   │   ├── model/                     # GNN & Model Inference Code
│   │   └── utils/                     # Smoothing filters for predictions
│   ├── frontend/                      # Web UI (index.html, style.css, script.js)
│   ├── training/                      # Model Training Scripts (.py) & Saved Models (.pth, .json)
│   ├── dataset/                       # Raw datasets (excluded from Git)
│   └── requirements.txt               # Python Dependencies
├── .gitignore                         # Python/ML ignores
└── README.md                          # Project Documentation
```

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Sreeing15/SignTranslate.git
cd SignTranslate/signbridge
```

### 2. Set up a virtual environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

*(Note: Ensure you have your `gnn_model.pth` and `xgb_model.json` generated and placed inside the `training/` folder before running the app).*

### 4. Run the Backend Server
```bash
cd backend
python app.py
```
The Flask server will start on `http://localhost:5000`.

### 5. Open the Frontend
Simply open the `signbridge/frontend/index.html` file in any modern web browser to start using the app. Allow the browser to access your webcam to begin translation.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Sreeing15/SignTranslate/issues).

---
*Developed by Sreeing15 & team.*
