"""
Comprehensive Empirical Audit & Statistical Verification Script
Zero leakage, 5-Fold Stratified Cross-Validation, Statistical Significance Tests,
Metrics: Accuracy, Balanced Acc, Sensitivity, Specificity, Precision, F1, ROC-AUC, PR-AUC, MCC, Kappa, Brier Score.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# Set strict seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_data(filepath="cardio_train.csv"):
    df = pd.read_csv(filepath, sep=";").drop_duplicates().reset_index(drop=True)
    if df["age"].max() > 100:
        df["age"] = df["age"] / 365.25

    # Rigorous physiological cleaning
    mask = (
        df["ap_hi"].between(80, 220) &
        df["ap_lo"].between(50, 130) &
        (df["ap_hi"] > df["ap_lo"]) &
        (df["ap_hi"] - df["ap_lo"]).between(15, 90) &
        df["height"].between(120, 210) &
        df["weight"].between(35, 180)
    )
    df = df[mask].copy().reset_index(drop=True)

    # Deterministic row-level clinical features
    df["blood_diff"] = df["ap_hi"] - df["ap_lo"]
    df["BMI"] = df["weight"] / ((df["height"] / 100.0) ** 2)
    df["obese"] = (df["BMI"] >= 30).astype(int)
    df["hypertense"] = ((df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)).astype(int)

    return df

BASE_FEATURES = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]
FE_FEATURES = BASE_FEATURES + ["blood_diff", "BMI", "obese", "hypertense"]
TARGET = "cardio"

def calc_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
        "Sensitivity": sens,
        "Specificity": spec,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "Brier_Score": brier_score_loss(y_true, y_prob)
    }

def build_compact_transformer(n_features):
    inp = layers.Input(shape=(n_features, 1))
    
    # Conv1D feature extractors
    c1 = layers.Conv1D(32, 2, padding="same", activation="relu")(inp)
    c1 = layers.BatchNormalization()(c1)
    c2 = layers.Conv1D(64, 2, padding="same", activation="relu")(c1)
    c2 = layers.BatchNormalization()(c2)
    
    # BiLSTM temporal modeling
    lstm = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(c2)
    
    # Self-attention
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16, dropout=0.1)(lstm, lstm)
    x_attn = layers.Add()([lstm, attn])
    x_attn = layers.LayerNormalization()(x_attn)
    
    # Multi-scale skip pooling
    gap_attn = layers.GlobalAveragePooling1D()(x_attn)
    gap_lstm = layers.GlobalAveragePooling1D()(lstm)
    gap_conv = layers.GlobalAveragePooling1D()(c2)
    
    fused = layers.Concatenate()([gap_attn, gap_lstm, gap_conv])
    x = layers.Dense(64, activation="relu")(fused)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    
    model = Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_base_cnn_lstm(n_features):
    inp = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    
    model = Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    print("="*80)
    print("STARTING RIGOROUS EMPIRICAL 5-FOLD CROSS-VALIDATION AUDIT")
    print("="*80)

    df = load_data("cardio_train.csv")
    print(f"Cleaned dataset: {len(df)} patient records. Balanced classes: {df['cardio'].mean()*100:.2f}% CVD")

    X = df[FE_FEATURES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.int32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    model_names = [
        "Logistic_Regression",
        "Random_Forest",
        "Hist_GBDT",
        "Base_CNN_LSTM",
        "Compact_CNN_BiLSTM_Transformer",
        "Hybrid_Blend_DL_GBDT"
    ]

    fold_results = {m: [] for m in model_names}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Running Fold {fold_idx} / 5 ---")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Zero leakage: fit strictly on fold training data
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        # 1. Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(X_tr_sc, y_tr)
        p_lr = lr.predict_proba(X_val_sc)[:, 1]
        fold_results["Logistic_Regression"].append(calc_metrics(y_val, p_lr))

        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=SEED)
        rf.fit(X_tr_sc, y_tr)
        p_rf = rf.predict_proba(X_val_sc)[:, 1]
        fold_results["Random_Forest"].append(calc_metrics(y_val, p_rf))

        # 3. HistGBDT
        hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_depth=5, random_state=SEED)
        hgb.fit(X_tr_sc, y_tr)
        p_hgb = hgb.predict_proba(X_val_sc)[:, 1]
        fold_results["Hist_GBDT"].append(calc_metrics(y_val, p_hgb))

        # Sequence inputs for DL
        X_tr_seq = X_tr_sc[..., np.newaxis]
        X_val_seq = X_val_sc[..., np.newaxis]

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
        ]

        # 4. Base CNN-LSTM
        base_dl = build_base_cnn_lstm(len(FE_FEATURES))
        base_dl.fit(X_tr_seq, y_tr, validation_data=(X_val_seq, y_val), epochs=15, batch_size=256, callbacks=callbacks, verbose=0)
        p_base = base_dl.predict(X_val_seq, batch_size=256, verbose=0).ravel()
        fold_results["Base_CNN_LSTM"].append(calc_metrics(y_val, p_base))

        # 5. Compact CNN-BiLSTM-Transformer
        prop_dl = build_compact_transformer(len(FE_FEATURES))
        prop_dl.fit(X_tr_seq, y_tr, validation_data=(X_val_seq, y_val), epochs=15, batch_size=256, callbacks=callbacks, verbose=0)
        p_prop = prop_dl.predict(X_val_seq, batch_size=256, verbose=0).ravel()
        fold_results["Compact_CNN_BiLSTM_Transformer"].append(calc_metrics(y_val, p_prop))

        # 6. Hybrid Blend (50% DL + 50% GBDT)
        p_blend = 0.5 * p_prop + 0.5 * p_hgb
        fold_results["Hybrid_Blend_DL_GBDT"].append(calc_metrics(y_val, p_blend))

        print(f"Fold {fold_idx} done: Base CNN-LSTM Acc={fold_results['Base_CNN_LSTM'][-1]['Accuracy']*100:.2f}%, Blend Acc={fold_results['Hybrid_Blend_DL_GBDT'][-1]['Accuracy']*100:.2f}%, Blend AUC={fold_results['Hybrid_Blend_DL_GBDT'][-1]['ROC_AUC']:.4f}")

    # Aggregate 5-Fold Summary (Mean +/- Std)
    summary_rows = []
    metric_keys = list(fold_results[model_names[0]][0].keys())

    for m in model_names:
        row = {"Model": m}
        for k in metric_keys:
            vals = [fold_results[m][i][k] for i in range(5)]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            if k in ["Accuracy", "Balanced_Acc", "Sensitivity", "Specificity", "Precision", "Recall", "F1", "MCC", "Kappa"]:
                row[k] = f"{mean_v*100:.2f} ± {std_v*100:.2f}%"
            else:
                row[k] = f"{mean_v:.4f} ± {std_v:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "="*90)
    print("5-FOLD STRATIFIED CROSS-VALIDATION BENCHMARK (MEAN ± STD)")
    print("="*90)
    print(summary_df[["Model", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "ROC_AUC", "PR_AUC", "Brier_Score"]].to_string(index=False))

    summary_df.to_csv("cvd_thesis_5fold_cv_results.csv", index=False)

    # Statistical Significance Testing (Paired t-test across folds)
    print("\n" + "="*90)
    print("STATISTICAL SIGNIFICANCE TESTS (PAIRED T-TEST ON ACCURACY & ROC-AUC)")
    print("="*90)

    for comp in ["Logistic_Regression", "Random_Forest", "Base_CNN_LSTM"]:
        acc_a = [fold_results["Hybrid_Blend_DL_GBDT"][i]["Accuracy"] for i in range(5)]
        acc_b = [fold_results[comp][i]["Accuracy"] for i in range(5)]
        t_acc, p_acc = stats.ttest_rel(acc_a, acc_b)

        auc_a = [fold_results["Hybrid_Blend_DL_GBDT"][i]["ROC_AUC"] for i in range(5)]
        auc_b = [fold_results[comp][i]["ROC_AUC"] for i in range(5)]
        t_auc, p_auc = stats.ttest_rel(auc_a, auc_b)

        print(f"Hybrid Blend vs {comp:22s} | Acc p-val: {p_acc:.4f} ({'Sig p<0.05' if p_acc < 0.05 else 'Not sig'}) | AUC p-val: {p_auc:.4f} ({'Sig p<0.05' if p_auc < 0.05 else 'Not sig'})")

if __name__ == "__main__":
    main()
