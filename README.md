# EEG Classification Using Kolmogorov–Arnold Networks (KAN)

This repository contains an end-to-end machine learning pipeline for **EEG-based classification of schizophrenia vs. healthy subjects**.  
The project integrates advanced **signal processing**, **time–frequency feature extraction**, **channel-wise PCA**, and a modern **Kolmogorov–Arnold Network (KAN)** neural architecture (PyKAN + PyTorch).

This project was developed as part of my bachelor’s thesis in Computer Science

---

## 🚀 Key Features

### 🧠 EEG Preprocessing Pipeline
- 4th-order **Butterworth bandpass filter** (0.5–45 Hz)  
- Segmentation of raw EEG into fixed windows (12 seconds @ 128 Hz)  
- **Short-Time Fourier Transform (STFT)**  
- **Channel-wise PCA** for dimensionality reduction  
- Produces a compact feature vector per EEG segment (~3000 features)

### 🤖 Kolmogorov–Arnold Network (KAN)
- Implemented using the **PyKAN** library  
- Spline-based neurons with:
  - Grid size = 8  
  - Spline order = 4  
- Network architecture:  
  **Input → 40 → 80 → 40 → Output (2 classes)**

### 📊 Evaluation
- Multiple repeated train/validation/test runs  
- Segment-level and subject-level classification  
- Metrics include:
  - Accuracy  
  - AUROC  
  - Precision  
  - Recall  

## 📂 Repository Structure

├── channelpca.py # EEG loading, filtering, STFT, PCA, and dataset generation
├── plotfinal.py # Main KAN training & evaluation pipeline
├── plot_file.py # Auxiliary plotting utilities
├── requirements.txt # Python dependencies for PyKAN + scientific stack
└── README.md # Project documentation


---

## ⚙️ Installation

You can install the project using **Conda (recommended)** or **pip**.

---

### 🔹 Option 1 — Conda Environment (Recommended)

```bash
conda create --name eegkan python=3.9.7
conda activate eegkan
pip install -r requirements.txt


All required versions (PyTorch 2.2.2, numpy 1.24.4, etc.) are included.

🔹 Option 2 — Manual Installation
pip install pykan
pip install numpy==1.24.4 matplotlib==3.6.2 scikit_learn==1.1.3 sympy==1.11.1 \
            torch==2.2.2 pandas==2.0.1 tqdm pyyaml seaborn

📄 Dataset Format

This project uses a public EEG dataset containing recordings from:

84 subjects (53 healthy, 31 schizophrenia)

16-channel EEG

128 Hz sampling rate

~60 seconds per subject

Stored in .eea format (shape: 16 × 7680 samples)

The dataset is publicly available at:

🔗 http://brain.bio.msu.ru/eeg_schizophrenia.htm

The pipeline automatically:

Segments each EEG file into 5 windows

Applies filtering

Computes STFT

Performs channel-wise PCA

Builds train/validation/test datasets

▶️ Running the Model

Example command:
python plotfinal.py \
    --healthy_dir healthy \
    --schizophrenia_dir schiz \
    --epochs 25 \
    --batch_size 25 \
    --num_runs 10

📊 Output Files

Results are saved automatically to:

./Results/Schizophrenia_ValidatedRuns/



