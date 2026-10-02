"""
CVD Thesis — CNN-BiLSTM-Transformer + SHAP + Clinical Digital Twin
Complete implementation pipeline for cardiovascular disease prediction,
ablation study, explainability, and digital twin simulation.
"""

import os
import sys
import glob
import json
import random
import argparse
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import shap
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description="Run CVD Thesis Deep Learning & Digital Twin Pipeline")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per model (default: 5)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training (default: 256)")
    parser.add_argument("--data-path", type=str, default="cardio_train.csv", help="Path to cardio_train.csv")
    parser.add_argument("--out-dir", type=str, default="cvd_thesis_results", help="Directory to save all outputs")
    parser.add_argument("--shap-samples", type=int, default=100, help="Number of samples to explain with SHAP")
    parser.add_argument("--shap-background", type=int, default=50, help="Number of background samples for SHAP")
    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_and_clean_data(data_path, out_dir):
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING & CLINICAL FEATURE ENGINEERING")
    print("="*80)

    # Kaggle dataset can use semicolon or comma
    try:
        df = pd.read_csv(data_path, sep=";")
        if "cardio" not in df.columns:
            df = pd.read_csv(data_path, sep=",")
    except Exception:
        df = pd.read_csv(data_path)

    print(f"Dataset loaded from: {data_path}")
    print(f"Raw shape: {df.shape}")

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert age from days to years
    if df["age"].max() > 100:
        df["age"] = df["age"] / 365.25

    # Paper-reported engineered features
    df["blood_diff"] = df["ap_hi"] - df["ap_lo"]
    df["BMI"] = df["weight"] / ((df["height"] / 100.0) ** 2)
    df["obese"] = (df["BMI"] >= 30).astype(int)
    df["hypertense"] = ((df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)).astype(int)

    # Physiological range cleaning: remove blood_diff outside [0, 80]
    before = len(df)
    df = df[df["blood_diff"].between(0, 80)].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    print(f"Physiologically invalid rows removed: {before - len(df)}")
    print(f"Cleaned dataset shape: {df.shape}")

    cleaned_path = os.path.join(out_dir, "cleaned_engineered_dataset.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset to: {cleaned_path}")

    # EDA Visualizations
    plt.figure(figsize=(10, 7))
    sns.heatmap(df.drop(columns=["id"], errors="ignore").corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap - Cardiovascular Dataset")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="cardio")
    plt.title("CVD Class Distribution")
    plt.xlabel("Cardio (0 = Normal, 1 = CVD)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=300)
    plt.close()

    return df

BASE_FEATURES = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]

FE_FEATURES = BASE_FEATURES + [
    "blood_diff", "BMI", "obese", "hypertense"
]

TARGET = "cardio"

def prepare_data(df, features, seed=42, test_size=0.20):
    X = df[features].astype(np.float32).values
    y = df[TARGET].astype(np.int32).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Scale strictly on train
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    X_train_seq = X_train[..., np.newaxis]
    X_test_seq = X_test[..., np.newaxis]

    return X_train, X_test, X_train_seq, X_test_seq, y_train, y_test, scaler

def conv_block(x, filters=64, kernel_size=3, dropout=0.20):
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = layers.Dropout(dropout)(x)
    return x

def build_cnn_lstm(input_shape):
    inp = layers.Input(shape=input_shape, name="clinical_input")
    x = conv_block(inp, 64, 3, 0.20)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(1, activation="sigmoid", name="cvd_probability")(x)
    return Model(inp, out, name="CNN_LSTM")

def build_cnn_bilstm(input_shape):
    inp = layers.Input(shape=input_shape, name="clinical_input")
    x = conv_block(inp, 64, 3, 0.20)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="CNN_BiLSTM")

def transformer_encoder(x, num_heads=4, key_dim=16, ff_dim=128, dropout=0.15):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ff = layers.Dense(ff_dim, activation="gelu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(x.shape[-1])(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

def build_cnn_bilstm_transformer(input_shape):
    inp = layers.Input(shape=input_shape, name="clinical_input")
    x = conv_block(inp, 64, 3, 0.20)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = transformer_encoder(x, num_heads=4, key_dim=16, ff_dim=128, dropout=0.15)
    x = transformer_encoder(x, num_heads=4, key_dim=16, ff_dim=128, dropout=0.15)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid", name="cvd_probability")(x)
    return Model(inp, out, name="CNN_BiLSTM_Transformer")

def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )
    return model

def make_callbacks(name, out_dir):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(out_dir, f"{name}.keras"),
            monitor="val_loss", save_best_only=True, verbose=0
        )
    ]

def train_and_evaluate(model_name, builder, features, df, epochs, batch_size, out_dir):
    print("\n" + "="*80)
    print(f"TRAINING MODEL: {model_name} (Epochs: {epochs}, Batch Size: {batch_size})")
    print("="*80)

    X_train, X_test, X_train_seq, X_test_seq, y_train, y_test, scaler = prepare_data(df, features)
    model = compile_model(builder(X_train_seq.shape[1:]))

    history = model.fit(
        X_train_seq, y_train,
        validation_split=0.20,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=make_callbacks(model_name, out_dir),
        verbose=1,
        shuffle=True
    )

    prob = model.predict(X_test_seq, batch_size=batch_size, verbose=0).ravel()
    pred = (prob >= 0.50).astype(int)

    metrics = {
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1": float(f1_score(y_test, pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_test, prob)),
        "PR_AUC": float(average_precision_score(y_test, prob))
    }

    print("\nClassification Report:")
    print(classification_report(y_test, pred, digits=4))
    print("Metrics Summary:", json.dumps(metrics, indent=2))

    # Save artifacts
    joblib.dump(scaler, os.path.join(out_dir, f"{model_name}_scaler.pkl"))
    np.save(os.path.join(out_dir, f"{model_name}_test_prob.npy"), prob)
    np.save(os.path.join(out_dir, f"{model_name}_test_y.npy"), y_test)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, f"{model_name}_metrics.csv"), index=False)

    return model, history.history, metrics, (X_test_seq, y_test, prob, pred, features)

def run_shap_analysis(model, test_data, features, out_dir, bg_size=50, sample_size=100):
    print("\n" + "="*80)
    print("STEP 4: SHAP EXPLAINABLE AI ANALYSIS")
    print("="*80)

    Xte, yte, prob, pred, _ = test_data
    n_bg = min(bg_size, len(Xte))
    n_explain = min(sample_size, len(Xte))

    background = Xte[:n_bg]
    explain_x = Xte[:n_explain]

    print(f"Computing SHAP values (Background: {n_bg}, Explain Samples: {n_explain})...")
    try:
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(explain_x)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)
        shap_values = np.squeeze(shap_values)
        explain_2d = np.squeeze(explain_x)

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        if explain_2d.ndim == 1:
            explain_2d = explain_2d.reshape(1, -1)

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        order = np.argsort(mean_abs)[::-1]

        shap_df = pd.DataFrame({
            "Feature": np.array(features)[order],
            "Mean_Absolute_SHAP": mean_abs[order]
        })
        shap_csv = os.path.join(out_dir, "proposed_SHAP_feature_importance.csv")
        shap_df.to_csv(shap_csv, index=False)
        print(f"SHAP feature importance saved to: {shap_csv}")
        print("\nTop Features by SHAP Importance:")
        print(shap_df.head(10).to_string(index=False))

        # Barplot
        plt.figure(figsize=(9, 6))
        topn = min(15, len(shap_df))
        sns.barplot(data=shap_df.iloc[:topn], x="Mean_Absolute_SHAP", y="Feature", palette="Blues_r")
        plt.title("Global SHAP Feature Importance - Proposed CNN-BiLSTM-Transformer")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "SHAP_global_importance.png"), dpi=300)
        plt.close()

        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, explain_2d, feature_names=features, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "SHAP_summary_plot.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print("SHAP visualizations generated successfully.")

    except Exception as e:
        print("SHAP explanation note:", repr(e))

def run_digital_twin_prototype(model, scaler, out_dir):
    print("\n" + "="*80)
    print("STEP 5: CLINICAL DIGITAL TWIN PROTOTYPE & WHAT-IF INTERVENTION")
    print("="*80)

    def risk_label(p):
        if p < 0.30:
            return "LOW"
        elif p < 0.60:
            return "MODERATE"
        elif p < 0.80:
            return "HIGH"
        return "VERY HIGH"

    def prepare_single_patient(patient_dict):
        row = pd.DataFrame([patient_dict]).copy()
        if row["age"].iloc[0] > 100:
            row["age"] = row["age"] / 365.25
        row["blood_diff"] = row["ap_hi"] - row["ap_lo"]
        row["BMI"] = row["weight"] / ((row["height"] / 100.0) ** 2)
        row["obese"] = (row["BMI"] >= 30).astype(int)
        row["hypertense"] = ((row["ap_hi"] >= 140) | (row["ap_lo"] >= 90)).astype(int)
        return row[FE_FEATURES].astype(np.float32)

    def predict_twin_state(patient_id, patient_dict):
        Xrow = prepare_single_patient(patient_dict)
        Xscaled = scaler.transform(Xrow).astype(np.float32)
        Xseq = Xscaled[..., np.newaxis]
        p = float(model.predict(Xseq, verbose=0)[0][0])
        return {
            "patient_id": patient_id,
            "clinical_data": patient_dict,
            "predicted_cvd_probability": round(p, 6),
            "risk_level": risk_label(p)
        }

    # Baseline Patient Profile: 55yo, Stage 2 Hypertension (150/95 mmHg), overweight, elevated cholesterol
    example_patient = {
        "age": 55,
        "gender": 1,
        "height": 165,
        "weight": 75,
        "ap_hi": 150,
        "ap_lo": 95,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }

    twin_v1 = predict_twin_state("P001", example_patient)
    print("\n[Digital Twin - Initial State]")
    print(json.dumps(twin_v1, indent=2))
    with open(os.path.join(out_dir, "P001_twin_initial.json"), "w") as f:
        json.dump(twin_v1, f, indent=2)

    # What-if Scenario: Antihypertensive therapy successfully regulates BP to 125/80 mmHg
    updated_patient = example_patient.copy()
    updated_patient["ap_hi"] = 125
    updated_patient["ap_lo"] = 80

    twin_v2 = predict_twin_state("P001", updated_patient)
    print("\n[Digital Twin - Post-Intervention State (BP reduced to 125/80)]")
    print(json.dumps(twin_v2, indent=2))
    with open(os.path.join(out_dir, "P001_twin_updated.json"), "w") as f:
        json.dump(twin_v2, f, indent=2)

    # Comparison summary
    comparison = pd.DataFrame([
        {
            "State": "Initial State",
            "Systolic_BP": example_patient["ap_hi"],
            "Diastolic_BP": example_patient["ap_lo"],
            "CVD_Probability": twin_v1["predicted_cvd_probability"],
            "Risk_Level": twin_v1["risk_level"]
        },
        {
            "State": "Post-Intervention / What-If",
            "Systolic_BP": updated_patient["ap_hi"],
            "Diastolic_BP": updated_patient["ap_lo"],
            "CVD_Probability": twin_v2["predicted_cvd_probability"],
            "Risk_Level": twin_v2["risk_level"]
        }
    ])
    comparison_csv = os.path.join(out_dir, "digital_twin_scenario_comparison.csv")
    comparison.to_csv(comparison_csv, index=False)
    print("\n[Scenario Comparison]")
    print(comparison.to_string(index=False))

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(42)

    print("="*80)
    print("STARTING CVD THESIS DEEP LEARNING & DIGITAL TWIN EXPERIMENT")
    print(f"Output Directory : {args.out_dir}")
    print(f"Epochs per model : {args.epochs}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Dataset path     : {args.data_path}")
    print("="*80)

    # Step 1: Data preparation
    df = load_and_clean_data(args.data_path, args.out_dir)

    # Step 2: Model Training & Ablation Study
    experiments = [
        ("01_Base_CNN_LSTM_No_FE", build_cnn_lstm, BASE_FEATURES),
        ("02_Base_CNN_LSTM_FE", build_cnn_lstm, FE_FEATURES),
        ("03_CNN_BiLSTM_FE", build_cnn_bilstm, FE_FEATURES),
        ("04_Proposed_CNN_BiLSTM_Transformer_FE", build_cnn_bilstm_transformer, FE_FEATURES)
    ]

    all_metrics = []
    test_sets = {}
    histories = {}
    trained_models = {}

    for name, builder, features in experiments:
        model, hist, metrics, test_data = train_and_evaluate(
            name, builder, features, df, args.epochs, args.batch_size, args.out_dir
        )
        all_metrics.append(metrics)
        histories[name] = hist
        test_sets[name] = test_data
        trained_models[name] = model

    # Step 3: Comparative Analysis & Visualizations
    results_df = pd.DataFrame(all_metrics)
    results_csv = os.path.join(args.out_dir, "ALL_MODEL_RESULTS.csv")
    results_df.to_csv(results_csv, index=False)

    print("\n" + "="*80)
    print("FINAL COMPARATIVE RESULTS TABLE")
    print("="*80)
    print(results_df.sort_values("Accuracy", ascending=False).to_string(index=False))

    # Accuracy Barplot
    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Accuracy", y="Model", palette="viridis")
    plt.xlim(0.50, 1.00)
    plt.title("Model Accuracy Comparison across Ablation Configurations")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "model_accuracy_comparison.png"), dpi=300)
    plt.close()

    # ROC & PR Curves
    plt.figure(figsize=(8, 6))
    for name, (Xte, yte, prob, pred, features) in test_sets.items():
        fpr, tpr, _ = roc_curve(yte, prob)
        auc = roc_auc_score(yte, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "ROC_comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, (Xte, yte, prob, pred, features) in test_sets.items():
        precision, recall, _ = precision_recall_curve(yte, prob)
        ap = average_precision_score(yte, prob)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curves")
    plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "PR_comparison.png"), dpi=300)
    plt.close()

    # Confusion Matrix for Proposed Model
    final_name = "04_Proposed_CNN_BiLSTM_Transformer_FE"
    Xte, yte, prob, pred, _ = test_sets[final_name]
    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "CVD"], yticklabels=["Normal", "CVD"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix - Proposed CNN-BiLSTM-Transformer")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "proposed_confusion_matrix.png"), dpi=300)
    plt.close()

    # Training & Validation Curves
    for name, h in histories.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(h["accuracy"], label="Train Accuracy")
        axes[0].plot(h["val_accuracy"], label="Val Accuracy")
        axes[0].set_title(f"{name} - Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        axes[1].plot(h["loss"], label="Train Loss")
        axes[1].plot(h["val_loss"], label="Val Loss")
        axes[1].set_title(f"{name} - Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{name}_learning_curves.png"), dpi=300)
        plt.close()

    # Step 4: SHAP Analysis for Proposed Model
    run_shap_analysis(
        trained_models[final_name],
        test_sets[final_name],
        FE_FEATURES,
        args.out_dir,
        bg_size=args.shap_background,
        sample_size=args.shap_samples
    )

    # Step 5: Clinical Digital Twin Prototype
    scaler = joblib.load(os.path.join(args.out_dir, f"{final_name}_scaler.pkl"))
    run_digital_twin_prototype(trained_models[final_name], scaler, args.out_dir)

    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"All artifacts, models, CSVs, and figures are in: {args.out_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
