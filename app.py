import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    yolo_model = YOLO('C:\\Users\\menuk\\Desktop\\Breast Cancer\\best.pt')
    yolo_model.model.to(device)
    return yolo_model, device

yolo_model, device = load_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    # Read the image
    frame = cv2.imread(image_path)
    
    # Resize the frame to (640, 640)
    frame_resized = cv2.resize(frame, (640, 640))

    # Convert to float and normalize
    frame_float = frame_resized.astype(np.float32) / 255.0

    # Prepare tensor
    frame_tensor = torch.from_numpy(frame_float).permute(2, 0, 1).unsqueeze(0).to(device)

    # Detect objects in the frame
    with torch.no_grad():
        results = yolo_model(frame_tensor)

    # Process detections
    detections_list = []
    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection.cpu().numpy()
                cv2.rectangle(frame_resized, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                cv2.putText(frame_resized, label, (int(xmin), int(ymin) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                detections_list.append({
                    'class': classes[int(cls[pos])],
                    'confidence': float(conf[pos]),
                    'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                })

    # Save the processed image
    output_path = os.path.join(UPLOAD_FOLDER, 'processed_image.jpg')
    cv2.imwrite(output_path, frame_resized * 255)

    return detections_list, output_path

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            try:
                detections, processed_image_path = process_image(filepath)
                return render_template('result.html', 
                                       original_image=f'/uploads/{filename}',
                                       processed_image='/uploads/processed_image.jpg', 
                                       detections=detections)
            except Exception as e:
                return jsonify({'error': str(e)})
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)