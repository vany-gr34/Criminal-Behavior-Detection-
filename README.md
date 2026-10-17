# 🛡️ Criminal Behavior Detection: Advanced Surveillance Intelligence

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Deep Learning](https://img.shields.io/badge/Domain-Deep%20Learning-green)](https://en.wikipedia.org/wiki/Deep_learning)
[![Accuracy](https://img.shields.io/badge/Max_Accuracy-90.04%25-brightgreen)](https://github.com/mohamedelattar15)

## 📌 Overview

**Criminal Behavior Detection** is a state-of-the-art Deep Learning framework designed for real-time automated security monitoring. By leveraging advanced **3D Convolutional Neural Networks (3D-CNNs)** and **Video Transformers**, the system identifies suspicious activities with high precision, classifying video streams into three critical categories:

*   🔴 **Violence**: Physical altercations and aggressive behavior.
*   🟠 **Theft**: Shoplifting, robbery, and unauthorized asset removal.
*   🟢 **Normal**: Safe, non-suspicious environmental activities.

---

## 🚀 Key Performance Benchmarks

The project evaluates 7 leading video action recognition architectures. Below is a comparative analysis of our results on the combined test set:

| Model Architecture | Test Accuracy | Categorization Strength |
| :--- | :---: | :--- |
| **MViT v2-T** | **90.04%** | 🔥 **State-of-the-Art Transformer** |
| **I3D (Inflated 3D)** | **87.22%** | High Feature Consistency |
| **Swin3D-T** | **84.76%** | Specialized Attention Gates |
| **X3D-M** | **81.30%** | Efficient Expansion |
| **R(2+1)D** | **78.45%** | Temporal-Spatial Decoupling |
| **TimeSformer** | **75.20%** | Divided Space-Time Attention |
| **SlowFast** | **71.75%** | Multi-Path Rate Fusion |

---

## 📊 Dataset Composition

Our robust training pipeline integrates three diverse datasets to ensure environmental generalization:

1.  **RWF-2000**: Specialized in violence detection in diverse lighting (Fight/Non-Fight).
2.  **UCF-Crime**: Large-scale dataset covering real-world robbery, stealing, and burglary events.
3.  **CAMNUVEOM**: Custom proprietary dataset for localized anomaly detection.

### Class Mapping
| Original Source | Unified Class |
| :--- | :--- |
| RWF [Fight] | **Violence** |
| UCF [Shoplifting, Robbery, Burglary] | **Theft** |
| CAMNUVEOM [Anomaly] | **Theft** |
| All [Normal] folders | **Normal** |

---

## 🛠️ Technical Stack

*   **Framework**: [PyTorch](https://pytorch.org/)
*   **Architectures**: [PyTorchVideo](https://pytorchvideo.org/) & [Torchvision](https://pytorch.org/vision/stable/models.html#video-classification)
*   **Vision Processing**: OpenCV & PIL
*   **Optimization**: AdamW with Cosine Annealing Learning Rate
*   **Data Augmentation**: Frame standardization (16 or 64 frames), 224x224 Resolution, ImageNet Normalization.

---

## 📂 Project Structure

```bash
Criminal Behavior Detection/
├── Codes/                    # Jupyter Notebooks for each model
│   ├── I3D Crime Detection Model.ipynb
│   ├── mvit-v2  Crime Detection Model.ipynb
│   ├── r-2-1-d-crime-detection-model.ipynb
│   ├── slowfast-crime-detection-model.ipynb
│   ├── swin3d  Crime Detection Model.ipynb
│   ├── timesformer-crime-detection-model.ipynb
│   └── x3d-crime-detection-model.ipynb
├── Criminal Behavior Detection _Papier.pdf  # Research Documentation
├── Datasets_info.pdf         # Detailed Dataset Analysis
└── README.md                 # Project Documentation
```

---

## ⚙️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/mohamedelattar15/-Criminal-Behavior-Detection-Advanced-Surveillance-Intelligence.git
    cd -Criminal-Behavior-Detection-Advanced-Surveillance-Intelligence
    ```

2.  **Environment Setup**:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install opencv-python pandas matplotlib seaborn tqdm
    ```

3.  **Run Models**:
    Open any notebook in the `Codes/` directory using Jupyter or Kaggle and update the dataset paths to your local directory.

---

## 📄 License & Citation

This project is part of a research initiative on intelligent safety. For detailed methodology, please refer to the included **Criminal Behavior Detection _Papier.pdf**.

---

<p align="center">
  Developed with ❤️ by <b>Mohamed Elattar</b> & Team
</p>