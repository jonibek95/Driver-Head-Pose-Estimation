# 🚗 Driver Head Pose Estimation  
Real-time Driver Monitoring System using FSA-Net (Face Semantic Attention Network)

This project builds a **real-time driver head pose estimation system** designed for **Driver Monitoring Systems (DMS)** such as drowsiness detection, distraction detection, and safe-driving assistance.

It uses a combination of:
- **FSA-Net (Capsule, Var, No-S) ensemble**
- **SSD-based face detection**
- **Smoothed yaw, pitch, roll estimation**
- **3D axis visualization over the driver’s face**
- **Orientation classification (Looking Left / Right / Up / Down)**  
- **Video output with visualization overlays**

---

## 📌 Features

### ✔️ **Real-time Driver Head Pose Estimation**
Predicts 3 orientation angles:
- **Yaw (left ↔ right)**
- **Pitch (up ↔ down)**
- **Roll (tilt)**

### ✔️ **FSA-Net Ensemble**
Uses 3 FSA-Net variants:
- **FSA-Net Capsule**
- **FSA-Net Var Capsule**
- **FSA-Net NoS Capsule**

Then averages their output → more stable & accurate predictions.

### ✔️ **Smooth Predictions**
A moving average filter (deque window) is used to avoid noisy angle jumps.

### ✔️ **Orientation Classification**
Automatically shows driver orientation:
- **Looking Left**
- **Looking Right**
- **Looking Up**
- **Looking Down**
- **Head Position is OK**

### ✔️ **On-Face 3D Axis Visualization**
Displays a 3D axis directly on the detected face region.

### ✔️ **Face Preview Window**
Small preview window with highlighted red border.

### ✔️ **Video Output**
Saves fully visualized video as:
output_head_pose.mp4

---

## 🎥 Demo  
* 
Example:

---

## 🛠 Installation

### 1️⃣ Create Conda environment
```bash
conda create -n fsa python=3.9
conda activate fsa
pip install -r requirements.txt

▶️ Run the system
Simply run:
python demo/Head_Estimation.py
