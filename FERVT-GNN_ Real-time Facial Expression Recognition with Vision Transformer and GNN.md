# FERVT-GNN: Real-time Facial Expression Recognition with Vision Transformer and GNN

This repository contains the implementation of a state-of-the-art Facial Expression Recognition (FER) system, named **FERVT-GNN** (Facial Expression Recognition with Vision Transformer and Graph Neural Network). The system uses a hybrid deep learning architecture for robust emotion classification and is designed for real-time deployment using **Flask** and **OpenCV**.

## 🌟 Features

*   **Hybrid Deep Learning Model:** Integrates a Vision Transformer (VT) for global feature extraction and a Graph Neural Network (GNN) for local, relational feature enhancement.
*   **Grid-Wise Attention (GWA):** Employs an attention mechanism to focus on salient facial regions.
*   **Training Pipeline:** Comprehensive training and evaluation scripts using PyTorch and standard datasets (e.g., FER2013/CK+).
*   **Real-time Deployment:** Includes a Flask application for live video processing, face detection, and emotion prediction via webcam.

## 🧠 Model Architecture Overview

The `FERVT_GNN` model is modularized into several key components:

| Component | File | Description |
| :--- | :--- | :--- |
| **GWA & Fusion** | `FERVT_GNN.py` | Extracts and fuses image features with an attention map to highlight expression-relevant areas. |
| **GNN Branch** | `graph.py` | Constructs a graph from the fused features and uses a Graph Attention Network (GAT) to model local relationships between facial regions. |
| **VT Branch** | `FERVT_GNN.py`, `transformer.py` | Uses a ResNet-based backbone to generate visual tokens, which are processed by a standard Transformer encoder for final classification. |
| **Training/Evaluation** | `main.py` | Handles data loading, augmentation, model training, and performance evaluation (e.g., confusion matrix, history plots). |

## 🛠️ Setup and Installation

### 1. Prerequisites

*   Python 3.8+
*   A trained model checkpoint (e.g., `best_model.pth`) is required for deployment.

### 2. Environment Setup

It is highly recommended to use a virtual environment:

```bash
python3 -m venv fer_env
source fer_env/bin/activate
```

### 3. Install Dependencies

The system requires several libraries, including PyTorch, PyTorch Geometric, Flask, and OpenCV.

```bash
# Install core dependencies
pip install torch torchvision numpy matplotlib scikit-learn Flask opencv-python

# Install PyTorch Geometric (PyG)
# Note: The installation of PyG is environment-specific. 
# For a standard CPU-only environment, the following command might work:
pip install torch_geometric

# If you encounter issues, please refer to the official PyTorch Geometric documentation
# for the correct installation command based on your PyTorch and CUDA versions.
```

## 🚀 Usage

### A. Model Training and Evaluation

The `main.py` script is designed for training the FERVT-GNN model on a suitable dataset (e.g., FER2013 or CK+).

1.  **Prepare Dataset:** Modify the `dataset_path` variable in `main.py` to point to your dataset directory.
2.  **Run Training:**
    ```bash
    python main.py
    ```
    This script will save the best model checkpoint to `./checkpoints/best_model.pth`.

### B. Real-time Deployment (Flask + OpenCV)

To run the real-time FER system, you need the following file structure:

```
fer-deployment/
├── FERVT_GNN.py
├── graph.py
├── transformer.py
├── app.py
├── best_model.pth           <-- **REQUIRED** Trained model checkpoint
└── templates/
    └── index.html
```

1.  **Place Checkpoint:** Ensure your trained `best_model.pth` is in the root directory.
2.  **Run the Flask App:**
    ```bash
    python app.py
    ```
3.  **Access the Application:** Open your web browser and navigate to the application address, typically:
    ```
    http://127.0.0.1:5000/
    ```
    The application will use your webcam to detect faces and display the real-time emotion prediction.

## 📂 File Structure

| File Name | Role |
| :--- | :--- |
| `FERVT_GNN.py` | Core model definition (Backbone, GWA, VTA, FERVT_GNN) |
| `graph.py` | GNN components (GraphConstructor, AttentionGNN, GraphPooling) |
| `transformer.py` | Transformer components (MultiHeadedSelfAttention, Block, Transformer) |
| `main.py` | Training and evaluation pipeline |
| `overallvisual.py` | Model visualization and analysis utilities |
| `app.py` | Flask application for real-time deployment |
| `templates/index.html` | Frontend HTML for video feed |
| `deployment_guide.md` | Detailed deployment instructions |
| `system_architecture.md` | Detailed model component analysis |

---
*Created by Manus AI*
