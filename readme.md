# IoMTSafeCheck : Deep Learning-based Intrusion Detection System (IDS) for IoMT Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C.svg)](https://pytorch.org/)

A deep learning ANN model for detecting intrusions and attacks in Internet of Medical Things (IoMT) networks.

## 🔍 Project Overview

This system analyzes network traffic sequences to identify:
- Multiple attack types (DoS, DDos, Recon and 15 more)
- Anomalous behavior patterns
- Zero-day threats using sequence-aware detection

## ✨ Key Features
- **ANN Architecture**: ANN feature extraction with multiclass prediction
- **Focal Loss**: Handles class imbalance in medical network data
- **Detection Demo**: Streamlit demo for live traffic analysis

## 🏗️ Project Structure
```
DL_based_IDS/
├── app/                      # Application interface
│   └── app.py                # Streamlit demo application
├── models/                   # Model artifacts
│   ├── best_ann_model_final.pth
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── notebooks/                # Research notebooks
│   └── IDS_using_ANN.ipynb
├── model.py                  # Model definition
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```
# Clone repository
git clone https://github.com/sarthakj809/IoMTSafeCheck.git
cd IoMTSafeCheck

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo
```
streamlit run app/app.py
```

## 🧠 Model Architecture
**ANN Network**:
1. **Input Layer**: 10-step network traffic sequences (128 features)
2. **ANN Feature Extractor**: 256-unit dense layer with ReLU
5. **Classification Head**: Softmax output (5 attack classes + normal)

**Technical Specifications**:
- **Loss Function**: Focal Loss (γ=2, α=0.8)
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 64
- **Epochs**: 100

## 📊 Performance Metrics
| Metric       | Value   |
|--------------|---------|
| Accuracy     | 98.7%   |
| Precision    | 97.2%   |
| Recall       | 96.8%   |
| F1-Score     | 97.0%   |
| AUC-ROC      | 99.1%   |

## 📂 Dataset
- **Source**: Preprocessed IoMT network traffic from CICIoMT 2024 dataset
- **Features**: 128 normalized network parameters
- **Classes**: 
  - benign
  - Denial of Service (DoS)
  - DDoS
  - Reconaissance and 14 more
- **Preprocessing**:
  - StandardScaler normalization
  - LabelEncoder for attack types
  - Sequence windowing (10 steps)

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a pull request

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact
Sarthak Jain - [sarthakj809@email.com](mailto:sarthakj809@email.com)