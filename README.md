---
title: Tri-Modal Health Screening
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Tri-Modal Health Screening System 🩺

A Tri-Modal AI architecture designed for advanced healthcare diagnostics, specifically engineered to predict lung **"Infiltration"**. This system maximizes precision and recall by fusing three distinct data modalities into a single, robust Regularized Fusion Network.

![image alt](https://github.com/aryanvaghsiya11-a11y/tri-modal-health-screening/blob/de87d462ab0fb634ce36197b81c683ab9a79b9a3/try-model-decription-image.jpg)

## 🧠 Architecture Overview

This project leverages a multi-input stacking ensemble approach to provide comprehensive medical predictions:

1. **Visual Data (Images/Scans):** Processed using **DenseNet121** to extract complex spatial features from medical imaging.
2. **Tabular Metadata (Clinical Records):** Analyzed using **XGBoost** to capture patterns within structured patient data.
3. **Sequential Vitals (Signal Data):** Modeled using a **Stacked Bi-LSTM** to interpret time-series and sequential physiological signals.
4. **Regularized Fusion Network:** The latent representations from all three base models are concatenated and passed through a fusion network, delivering a highly accurate final prediction.

## 📂 Repository Structure

* `Model/`: Directory for storing trained model weights, scalers, and ensemble configurations.
* `samples/`: Sample data inputs (visual, tabular, and signal) for testing the inference pipeline.
* `static/` & `templates/`: Assets and HTML templates for the web-based user interface.
* `app.py`: The main web application script to serve the model and handle user requests.
* `inference.py`: Core logic for preprocessing inputs, loading the ensemble, and generating predictions.
* `tri-modal-by-aryan.ipynb`: Comprehensive Jupyter Notebook containing exploratory data analysis (EDA), model training, and evaluation metrics.
* `tri-modal-stripped.py`: Streamlined Python script containing the core model architecture.
* `test_*.py` files: Various testing scripts (`test_samples.py`, `test_upload.py`, `test_gradcam_variance.py`) to validate model explainability (Grad-CAM), file uploads, and inference accuracy.
* `requirements.txt`: List of Python dependencies required to run the project.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/aryanvaghsiya11-a11y/tri-modal-health-screening.git](https://github.com/aryanvaghsiya11-a11y/tri-modal-health-screening.git)
   cd tri-modal-health-screening
Install the required dependencies:

Bash
pip install -r requirements.txt
Running the Application
To start the web interface for the health screening system, run:

Bash
python app.py
The application will be accessible via your local host (typically http://localhost:5000 or http://127.0.0.1:5000).

🧪 Testing & Explainability
This repository includes scripts to test the robustness of the system and ensure the AI's decisions are interpretable by medical professionals:

Run python test_samples.py to verify the pipeline with default sample data.

Run python test_gradcam_variance.py to generate visual explanations (Grad-CAM heatmaps) that highlight which regions of the visual data influenced the DenseNet121 model's predictions.

🛠️ Built With
Deep Learning Frameworks: TensorFlow / Keras, PyTorch (Depending on notebook implementation)

Machine Learning: XGBoost, Scikit-Learn

Computer Vision: OpenCV

Web Framework: Flask / FastAPI
