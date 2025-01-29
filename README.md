# **🚀 Breast Cancer Detection using YOLOv8**  

🔬 **Early detection of breast cancer can save lives!** This AI-powered detection system uses **YOLOv8**, **Flask**, and **OpenCV** to identify potential cancerous regions in medical images.  

---

## **🩺 Why This Project?**  
🔍 **Fast & Accurate Detection** – Leveraging YOLOv8 for real-time analysis.  
🌐 **Web-Based Interface** – Upload images and get instant results.  
🎯 **Custom Training Support** – Train the model on new datasets.  
🛠 **Built with Open-Source Tech** – Python, Flask, OpenCV, and PyTorch.  

---

## **⚙️ Features**  
✅ AI-powered breast cancer detection  
✅ Real-time image processing  
✅ Flask-based web interface  
✅ Easy model training with YOLOv8  
✅ Supports multiple image formats (JPG, PNG, JPEG)  

---

## **📦 Tech Stack**  
🔹 **Deep Learning**: YOLOv8 (Ultralytics)  
🔹 **Backend**: Flask (Python)  
🔹 **Frontend**: HTML, CSS  
🔹 **Data Processing**: OpenCV, NumPy  
🔹 **Deployment**: Local/Cloud  

---

## **🛠 Setup & Installation**  

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/your-username/breast-cancer-detection-yolov8.git
cd breast-cancer-detection-yolov8
```

2️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```

3️⃣ **Run the application**  
```bash
python app.py
```
🔗 Open **`http://127.0.0.1:5000`** in your browser and upload an image for detection.

---

## **📊 Training the Model**  
If you want to train your **own YOLOv8 model**, follow these steps:  
```bash
yolo task=detect mode=train model=yolov8x.pt data=data.yaml epochs=50 imgsz=640 plots=True
```
👉 Save the trained weights inside the `/models` folder.

---

## **📜 License**  
This project is **open-source** and available under the **MIT License**.

### ⭐ *Star this repository if you like our project!* ⭐
