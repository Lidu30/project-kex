# EEG Classification with Kolmogorov-Arnold Networks (KAN)

This repository implements an end-to-end machine learning pipeline for **EEG-based classification of schizophrenia vs. healthy subjects**.  
The project combines advanced **signal processing** with a **Kolmogorov-Arnold Network (KAN)** model, showcasing data preprocessing, feature engineering, and deep learning integration in PyTorch.


## 🚀 Project Highlights
- **Preprocessing pipeline**  
  - Bandpass filtering with a 4th-order Butterworth filter (0.5–45 Hz)  
  - Short-Time Fourier Transform (STFT) for time–frequency representation  
  - Channel-wise PCA for dimensionality reduction  
- **Modeling**  
  - Lightweight KAN architecture (PyKAN + PyTorch)  
  - Adaptive splines with grid size 8 and spline order 4  
- **Evaluation**  
  - Repeated train/validation/test splits for robust performance estimation  
  - Metrics: Accuracy, AUROC, Precision, Recall, F1-score, Specificity  
- **Results**  
  - ~81.7% accuracy at **segment level**  
  - ~86.5% accuracy at **subject level**


## 📂 Repository Structure
├── channelpca.py # Data loading, preprocessing, STFT, PCA, dataset creation
├── plotfinal.py # Training & evaluation loop with repeated runs
├── plot_file.py #

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt

