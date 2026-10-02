# Cardiovascular Disease Identification: Hybrid CNN-BiLSTM-Transformer + SHAP + Clinical Digital Twin

This repository contains the complete implementation, ablation experiments, explainability analysis (XAI), and clinical Digital Twin prototype for Cardiovascular Disease (CVD) identification.

This study builds upon and extends the foundational research published in *Informatics in Medicine Unlocked (Elsevier, 2023)*: *"Cardiovascular disease identification using a hybrid CNN-LSTM model with explainable AI"*.

---

## Architecture Overview

```
                      Raw Clinical Input (cardio_train.csv)
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │ 1. Clinical Data Cleaning & Feature Engineering      │
             │   • Convert age (days ➔ years)                      │
             │   • Pulse pressure (blood_diff = ap_hi - ap_lo)      │
             │   • Body Mass Index (BMI = weight / (height/100)^2) │
             │   • Binary Obesity (BMI ≥ 30)                       │
             │   • Binary Hypertension (BP ≥ 140/90 mmHg)          │
             │   • Physiological filtering (0 ≤ blood_diff ≤ 80)   │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │ 2. Leakage-Free Stratified Preprocessing            │
             │   • Stratified 80/20 train/test split               │
             │   • MinMaxScaler strictly fitted on training set    │
             │   • Temporal sequence expansion: (samples, N, 1)    │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │ 3. Proposed Deep Learning Architecture              │
             │   • Conv1D Block (Local feature extraction + BN)    │
             │   • Bidirectional LSTM (Sequential context)         │
             │   • Dual Multi-Head Self-Attention (Global context) │
             │   • GlobalAveragePooling1D + Dense Classification   │
             └──────────────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
    ┌───────────────────────────────┐       ┌───────────────────────────────┐
    │ 4. Explainable AI (SHAP)      │       │ 5. Clinical Digital Twin      │
    │   • GradientExplainer attributions│   │   • Dynamic patient state     │
    │   • Global importance ranking │       │   • What-if intervention test │
    └───────────────────────────────┘       └───────────────────────────────┘
```

---

## Comparative Ablation Study Results

Experiments conducted on the Kaggle Cardiovascular Disease dataset (70,000 records, cleaned to 68,037 instances):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`01_Base_CNN_LSTM_No_FE`** | 73.38% | 75.43% | **67.99%** | **71.52%** | **0.7977** | **0.7815** |
| **`04_Proposed_CNN_BiLSTM_Transformer_FE`** | **73.12%** | **75.91%** | **66.37%** | **70.82%** | **0.7954** | **0.7761** |
| **`02_Base_CNN_LSTM_FE`** | 72.79% | 78.55% | 61.39% | 68.92% | 0.7957 | 0.7770 |
| **`03_CNN_BiLSTM_FE`** | 72.62% | **79.55%** | 59.61% | 68.15% | 0.7963 | 0.7796 |

All trained weights, scalers, and predictions are available in `cvd_thesis_results/`.

---

## Explainable AI (SHAP) Insights

SHAP GradientExplainer analysis on the proposed model reveals that clinical engineered biomarkers provide the highest predictive signal:

| Rank | Clinical Biomarker | Mean Absolute SHAP | Clinical Significance |
|:---:|---|:---:|---|
| **1** | `hypertense` | **0.0957** | Chief risk factor for arterial wall damage |
| **2** | `blood_diff` | **0.0462** | Pulse pressure marker for arterial stiffening |
| **3** | `obese` | **0.0397** | Metabolic risk indicator |
| **4** | `age` | **0.0379** | Age-related cardiovascular risk |
| **5** | `cholesterol` | **0.0240** | Atherosclerosis determinant |

---

## Clinical Digital Twin Prototype

The Digital Twin maintains a digital representation of a patient: clinical attributes $\rightarrow$ predicted probability $\rightarrow$ risk stratification (`LOW`, `MODERATE`, `HIGH`, `VERY HIGH`).

### What-If Simulation:
- **Patient Initial State**: 55-year-old, Stage 2 Hypertension (150/95 mmHg), overweight, elevated cholesterol.
  - Predicted CVD Risk: **81.42% (VERY HIGH)**
- **Post-Therapeutic Intervention**: Antihypertensive therapy regulates blood pressure to 125/80 mmHg.
  - Predicted CVD Risk: **45.46% (MODERATE)**
  - **Net Absolute Risk Reduction**: **-35.96%**

---

## Quickstart

### 1. Installation
Clone the repository and set up the Python environment:
```bash
git clone https://github.com/vikaxxh/CNN_BILSTM_TRANSFORMER-.git
cd CNN_BILSTM_TRANSFORMER-

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn shap joblib
```

### 2. Run the Full Pipeline via CLI
```bash
python run_pipeline.py --epochs 100 --batch-size 256 --out-dir cvd_thesis_results
```

For quick verification:
```bash
python run_pipeline.py --epochs 5 --batch-size 256
```

### 3. Run via Jupyter Notebook
Open [`CVD_Thesis_CNN_BiLSTM_Transformer_DigitalTwin (2).ipynb`](CVD_Thesis_CNN_BiLSTM_Transformer_DigitalTwin%20(2).ipynb) in JupyterLab, VS Code, or Google Colab. The notebook auto-detects local vs Colab runtime paths.

---

## Repository Structure

```
├── cardio_train.csv                                    # Kaggle Cardiovascular Disease Dataset
├── CVD_Thesis_CNN_BiLSTM_Transformer_DigitalTwin (2).ipynb # Interactive Jupyter Notebook
├── run_pipeline.py                                     # Reproducible CLI training pipeline
├── elsiver.pdf                                         # Reference paper (Elsevier IMU 2023)
├── cvd_thesis_results/                                 # Generated evaluation artifacts
│   ├── ALL_MODEL_RESULTS.csv                           # Comparative metrics table
│   ├── cleaned_engineered_dataset.csv                  # Preprocessed dataset with engineered features
│   ├── proposed_SHAP_feature_importance.csv            # Quantitative SHAP ranking
│   ├── digital_twin_scenario_comparison.csv            # Pre vs Post intervention comparison
│   ├── P001_twin_initial.json                          # Initial twin state JSON
│   ├── P001_twin_updated.json                          # Post-intervention twin state JSON
│   ├── *.keras                                         # Saved model weights
│   ├── *.pkl                                           # Scalers for each architecture
│   └── *.png                                           # ROC, PR, learning curves, and SHAP plots
└── README.md
```
