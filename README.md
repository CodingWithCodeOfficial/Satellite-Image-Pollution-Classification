# Satellite Image Pollution Classification

A deep-learning project that uses convolutional neural networks (CNNs) to classify satellite images (extracted from the STAC API) as polluted vs. non-polluted. Includes dataset preprocessing, model training, evaluation graphs, and a full inference pipeline for real-world environmental monitoring applications.

## 🚀 Overview

This repository implements a complete machine learning pipeline for detecting pollution in satellite imagery. By leveraging convolutional neural networks (CNNs), the project processes high-resolution satellite images sourced from the SpatioTemporal Asset Catalog (STAC) API to perform binary classification: **polluted** vs. **non-polluted**.

The system is designed for scalability and real-world use, making it suitable for environmental monitoring, urban planning, and climate research. It includes:

- Automated data extraction from satellite catalogs.
- Preprocessing scripts to prepare images for training.
- CNN model training and evaluation with performance visualizations.
- An inference pipeline for classifying new images.
- A Django-based web application for user-friendly interaction, allowing image uploads and real-time predictions.

This project demonstrates practical applications of deep learning in remote sensing and geospatial analysis, emphasizing explainable AI through evaluation metrics and graphs.

## 🛠️ Key Features

- 🌍 **STAC API Integration**: Automatically queries and downloads satellite imagery (e.g., from Sentinel-2 or similar collections) using geographic bounding boxes.
- 🖼️ **Dataset Preprocessing**: Handles image cleaning, resizing, augmentation, and weak labeling (e.g., using edge detection for haze proxies).
- 🧠 **CNN-Based Classification**: Trains small, efficient convolutional neural networks for binary pollution detection.
- 📊 **Model Evaluation**: Generates training/validation curves, confusion matrices, prediction galleries, and Grad-CAM visualizations for interpretability.
- ⚙️ **Inference Pipeline**: Full end-to-end workflow for classifying new satellite images, including batch processing.
- 🌐 **Django Web Interface**: A simple web app (`ml_app`) for uploading images, running predictions, and viewing results.
- 💾 **Pre-Trained Models**: Includes ready-to-use Keras models for harsh pollution and haze detection.
- 📈 **Visual Outputs**: Auto-saves plots for loss/accuracy, misclassifications, and attention maps.

## 🧰 Tech Stack

- **Programming Language**: Python (96% of the codebase)
- **Deep Learning Framework**: TensorFlow / Keras (for CNN model definition, training, and inference)
- **Web Framework**: Django (for the interactive web application)
- **Frontend**: HTML, CSS, JavaScript (for templates and basic interactivity in the web app)
- **Data Processing**: NumPy, Matplotlib (for arrays, visualizations); inferred libraries like OpenCV/PIL for image handling and rasterio/geopandas for geospatial data.
- **Other Tools**: Git for version control; Shell/PowerShell scripts for automation (e.g., `run_server.sh`).
- **Models & Files**: `.keras` format for saved models; SQLite (`db.sqlite3`) for Django's database.

The stack is kept lightweight to ensure fast development and deployment, focusing on core ML tasks while providing a user-facing interface.

I worked on this system alongside a project partner, coordinating our roles to design, train, and validate the pollution‑classification model.

**Installation details will be added later**
