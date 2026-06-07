🌾 Rice Leaf Disease Classification (Full Stack ML Website)

# Rice Leaf Disease Detection 🌱

🔗 Live Demo: https://rice-leaf-disease-detection-phi.vercel.app

A full-stack AI application for detecting rice leaf diseases using CNN and Transfer Learning.

📌 Project Overview

This project is an end-to-end Deep Learning web application that classifies rice leaf images into different disease categories using a Convolutional Neural Network (CNN).
Users can upload an image of a rice leaf through a web interface, and the system predicts the disease type in real-time.

🎯 Problem Statement

Rice crops are highly vulnerable to leaf diseases such as:
Bacterial Leaf Blight
Brown Spot
Leaf Blast
Healthy
Manual disease identification requires expert knowledge and is time-consuming.
This project automates the detection process using Artificial Intelligence.

🧠 Model Details

Model Type: Convolutional Neural Network (CNN)
Framework: TensorFlow / Keras
Input Size: (e.g., 224x224)
Output Classes:
bacterial_leaf_blight
brown_spot
healthy
leaf_blast
Training Method: Supervised Learning
Data Split: Train / Validation

🏗️ Tech Stack
🔹 Backend
  Python
  Flask
  TensorFlow / Keras
🔹 Frontend
  React.js
  HTML
  CSS
  JavaScript
🔹 Tools
  VS Code
  Git
  GitHub

🔄 System Architecture

User Flow:
User Uploads Image
→ React Frontend
→ Flask API
→ CNN Model Prediction
→ Result Returned
→ Displayed on UI

📂 Project Structure
project/
│
├── backend/
│   ├── app.py
│   ├── train.py
│   ├── rice_model.h5
│   └── venv/
│
├── frontend/
│   ├── src/
│   ├── package.json
│
├── .gitignore
├── README.md

🚀 How To Run Locally

1️⃣ Clone Repository
git clone https://github.com/yourusername/rice-leaf-disease-classification.git
2️⃣ Start Backend
cd backend
venv\Scripts\activate
python app.py
Server runs on:
http://127.0.0.1:5000
3️⃣ Start Frontend
cd frontend
npm install
npm start
App runs on:
http://localhost:3000

📊 Features

Image Upload Support
Real-time Disease Prediction
Clean UI
Backend API Integration
Model Training Script Included
Full Stack Architecture

🔮 Future Improvements

📱 Develop a mobile application to allow farmers to easily detect rice leaf diseases directly from their smartphones.
🤖 Integrate an AI chatbot assistant to help farmers with disease information, prevention tips, and treatment suggestions.
🌐 Add multilingual support to improve accessibility for farmers from different regions and language backgrounds.

👨‍💻 Author
Prince Kumar

