☀️ Solar Panel Fault Detection & Cleaning Classification

📌 Project Overview

This project is a deep learning-based web application that detects the condition of solar panels using image input.

The system classifies solar panel images into different categories such as:

Clean

Dusty

Faulty

Other defect types (based on training data)

The goal is to help identify panels that require cleaning or maintenance in order to improve solar energy efficiency.

🎯 Objective

Solar panels often lose efficiency due to:

Dust accumulation

Surface damage

Structural faults

Manual inspection is time-consuming and inefficient.

This project automates the detection process using computer vision and deep learning techniques.

🧠 Model Details

Framework: TensorFlow & Keras

Architecture: VGG16 (Transfer Learning)

Custom classification layers added on top

Trained on labeled solar panel image dataset

The trained model is saved as:

solar_models.h5
⚙️ How It Works

User uploads a solar panel image through the web interface.

The image is resized and preprocessed.

The trained CNN model predicts the condition.

The result is displayed on the screen.

🏗️ Project Structure
solar_panel_cleaning/
│
├── solar_panel/
│   ├── app.py
│   ├── requirements.txt
│   ├── solar_models.h5
│
└── README.md
🌐 Web Application

The project includes a Flask-based web application where users can:

Upload images

Get instant classification results

Determine whether cleaning or maintenance is required

🚀 Features

Image-based solar panel condition detection

Transfer learning using VGG16

Web interface built with Flask

Deployable on cloud platforms

Automated prediction system

🛠️ Technologies Used

Python

Flask

TensorFlow

Keras

NumPy

Pillow
