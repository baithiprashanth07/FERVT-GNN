import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, render_template, Response
from torchvision import transforms

# --- Import Model Components from User's Files ---
# Assuming all model files (FERVT_GNN.py, graph.py, transformer.py) are in the same directory
# In a real deployment, these would be copied or packaged.
from FERVT_GNN import FERVT_GNN

# --- Configuration ---
app = Flask(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pth' # Placeholder: User must provide a trained model file
INPUT_SIZE = (64, 64)
EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

# --- Global Model and Preprocessor ---
model = None
face_cascade = None
preprocess = None

def load_model_and_assets():
    """Loads the FER model and the Haar Cascade for face detection."""
    global model, face_cascade, preprocess
    
    # 1. Load Face Detector (Using OpenCV's built-in Haar Cascade for simplicity)
    # User might need to ensure this XML file is available
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model checkpoint not found at {MODEL_PATH}. Model will not be loaded.")
        print("Please place your trained 'best_model.pth' file in the application directory.")
        return

    # 2. Initialize and Load Model
    try:
        model = FERVT_GNN(device=DEVICE, use_gnn=True)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        model = None # Set to None if loading fails
        return

    # 3. Define Preprocessing Transforms (Matching main.py's test/val transforms)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load assets when the application starts
with app.app_context():
    load_model_and_assets()

def predict_emotion(face_img):
    """Performs emotion prediction on a single face image."""
    if model is None:
        return "Model Not Loaded"

    try:
        # Convert BGR to RGB (OpenCV default is BGR)
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Apply PyTorch preprocessing
        input_tensor = preprocess(rgb_img).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probabilities).item()
            
        emotion = EMOTION_LABELS.get(pred_idx, "Unknown")
        confidence = probabilities[0, pred_idx].item()
        
        return f"{emotion} ({confidence*100:.2f}%)"
    
    except Exception as e:
        # print(f"Prediction Error: {e}")
        return "Prediction Error"

def generate_frames():
    """Video streaming generator function."""
    camera = cv2.VideoCapture(0) # Use 0 for default webcam

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Face Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Scale factor and min neighbors can be tuned for performance/accuracy
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Predict emotion
            emotion_text = predict_emotion(face_roi)
            
            # Draw bounding box and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in Motion JPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Home page to display the video feed."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route to stream the video frames."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Flask is run in a separate thread, so the global assets are loaded via app_context
    # The user should run this script in their local environment where a webcam is available
    # For sandbox testing, we'll use a public port
    app.run(host='0.0.0.0', port=5000, debug=True)
