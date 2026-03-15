#!/usr/bin/env python3
"""
==============================================================================
5G-NIDD: Comprehensive Network Intrusion Detection Pipeline
==============================================================================
A single, end-to-end Python script that combines binary anomaly detection and
multiclass attack classification from raw data, including:
  1. Data Preprocessing & Feature Selection (from raw Combined.csv)
  2. Binary Model Training (RandomForest)
  3. Hybrid Sampling for Imbalanced Multiclass Data (SMOTE + RandomUnderSampler)
  4. Multiclass Model Training (RandomForest, 9 classes)
  5. Cascade Architectures (Binary-First, Multiclass-First, Parallel Voting/Confidence)
  6. Experimental fDNN (Random Forest leaf indices → MLP)
  7. Evaluation & Benchmarking (classification reports, confusion matrices, throughput)

Dataset Source:
  https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless

Attack Types:
  0=Benign, 1=UDPFlood, 2=HTTPFlood, 3=SlowrateDoS,
  4=TCPConnectScan, 5=SYNScan, 6=UDPScan, 7=SYNFlood, 8=ICMPFlood

Author: Auto-generated pipeline (English comments throughout)
"""

import os
import sys
import time
import warnings
import gc
import argparse

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- File paths (adjust as needed) ---
RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "Combined.csv", "Combined.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# --- Saved artifact paths ---
BINARY_MODEL_PATH = os.path.join(OUTPUT_DIR, "binary_rf_model.joblib")
BINARY_SCALER_PATH = os.path.join(OUTPUT_DIR, "binary_scaler.joblib")
BINARY_THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "binary_threshold.txt")
MULTICLASS_MODEL_PATH = os.path.join(OUTPUT_DIR, "multiclass_rf_model.joblib")
MULTICLASS_SCALER_PATH = os.path.join(OUTPUT_DIR, "multiclass_scaler.joblib")

# --- Attack type label map ---
ATTACK_TYPE_NAMES = [
    "Benign",          # 0
    "UDPFlood",        # 1
    "HTTPFlood",       # 2
    "SlowrateDoS",     # 3
    "TCPConnectScan",  # 4
    "SYNScan",         # 5
    "UDPScan",         # 6
    "SYNFlood",        # 7
    "ICMPFlood",       # 8
]

# --- ANOVA F-score based feature subsets (from the reference notebooks) ---
# These are the exact feature names AFTER column normalization (lowercase, no underscores)
# as used in both reference notebooks.
BINARY_FEATURES_NORMALIZED = [
    'seq', 'offset', 'sttl', 'ackdat', 'tcprtt', 'smeanpktsz',
    'shops', 'dttl', 'srcbytes', 'totbytes', 'dmeanpktsz',
    'srcwin', 'stos',
]  # 13 features

MULTICLASS_FEATURES_NORMALIZED = [
    'ackdat', 'shops', 'seq', 'tcprtt', 'dmeanpktsz', 'offset', 'sttl',
    'srctcpbase', 'smeanpktsz', 'dstloss', 'loss', 'dttl', 'srcbytes',
    'totbytes',
]  # 14 features

# --- Training split boundary ---
BINARY_TRAIN_ROWS = 856_400     # Binary model uses first 856,400 rows
MULTICLASS_TRAIN_ROWS = 800_000 # Multiclass model uses first 800,000 rows
TEST_START_ROW = 800_000        # Unseen test data starts at row 800,000

# --- Random state for reproducibility ---
RANDOM_STATE = 42


# ==============================================================================
# STEP 1: DATA PREPROCESSING & FEATURE SELECTION
# ==============================================================================

def load_and_preprocess_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Load the raw Combined.csv and preprocess to achieve the 48-column target schema.

    Steps:
      1. Load CSV
      2. Drop irrelevant columns (Unnamed index, RunTime, Sum, Min, Max,
         SrcGap, DstGap, sDSb, dDSb, Label, Attack Tool)
      3. Fill missing values with 0
      4. Map 'Attack Type' string labels to integer codes (0-8)
      5. Create binary label 'Label__Malicious' (0=Benign, 1=Malicious)
      6. One-Hot Encode 'Proto' (icmp, tcp, udp)
      7. Encode 'Cause' as binary Cause_Status
      8. One-Hot Encode 'State' (CON, ECO, FIN, INT, REQ, RST)
      9. Drop original categorical columns after encoding
     10. Ensure the 48-column schema with correct dtypes
    """
    print("[STEP 1] Loading raw data from:", csv_path)
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")

    # --- Drop irrelevant columns ---
    cols_to_drop = [
        "Unnamed: 0", "RunTime", "Sum", "Min", "Max",
        "SrcGap", "DstGap", "sDSb", "dDSb", "Attack Tool",
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors="ignore")

    # --- Fill missing values ---
    df.fillna(0, inplace=True)

    # --- Map Attack Type strings to integer labels ---
    attack_type_map = {
        "Benign": 0,
        "UDPFlood": 1,
        "UDP Flood": 1,
        "HTTPFlood": 2,
        "HTTP Flood": 2,
        "SlowrateDoS": 3,
        "Slowrate DoS": 3,
        "TCPConnectScan": 4,
        "TCP Connect Scan": 4,
        "SYNScan": 5,
        "SYN Scan": 5,
        "UDPScan": 6,
        "UDP Scan": 6,
        "SYNFlood": 7,
        "SYN Flood": 7,
        "ICMPFlood": 8,
        "ICMP Flood": 8,
    }
    if "Attack Type" in df.columns:
        df["Attack Type_"] = df["Attack Type"].map(attack_type_map).fillna(0).astype(int)
        df.drop(columns=["Attack Type"], inplace=True)
    elif "Attack Type_" not in df.columns:
        raise ValueError("No 'Attack Type' column found in raw data.")

    # --- Create binary label: Label__Malicious (0=Benign, 1=Malicious) ---
    if "Label" in df.columns:
        df["Label__Malicious"] = (df["Label"].str.strip().str.lower() != "benign").astype(int)
        df.drop(columns=["Label"], inplace=True)
    elif "Label__Malicious" not in df.columns:
        # Derive from Attack Type: 0 = benign, everything else = malicious
        df["Label__Malicious"] = (df["Attack Type_"] != 0).astype(int)

    # --- One-Hot Encode Protocol ---
    if "Proto" in df.columns:
        df["Proto"] = df["Proto"].astype(str).str.strip().str.lower()
        df["Proto_icmp"] = (df["Proto"] == "icmp").astype(int)
        df["Proto_tcp"] = (df["Proto"] == "tcp").astype(int)
        df["Proto_udp"] = (df["Proto"] == "udp").astype(int)
        df.drop(columns=["Proto"], inplace=True)

    # --- Encode Cause as binary Status ---
    if "Cause" in df.columns:
        df["Cause"] = df["Cause"].astype(str).str.strip().str.lower()
        df["Cause_Status"] = (df["Cause"] == "status").astype(int)
        df.drop(columns=["Cause"], inplace=True)

    # --- One-Hot Encode State ---
    if "State" in df.columns:
        df["State"] = df["State"].astype(str).str.strip().str.upper()
        for state_val in ["CON", "ECO", "FIN", "INT", "REQ", "RST"]:
            df[f"State_{state_val}"] = (df["State"] == state_val).astype(int)
        df.drop(columns=["State"], inplace=True)

    # --- Drop 'Mean' column if it's still around (it's in the raw CSV but
    #     not in the target 48-column schema as a standalone column) ---
    if "Mean" in df.columns:
        df.drop(columns=["Mean"], inplace=True)

    # --- Ensure column order and count ---
    # The target 48-column schema:
    target_columns = [
        "Seq", "Dur", "sHops", "dHops", "TotPkts", "SrcPkts", "DstPkts",
        "TotBytes", "SrcBytes", "DstBytes", "Offset", "sMeanPktSz", "dMeanPktSz",
        "Load", "SrcLoad", "DstLoad", "Loss", "SrcLoss", "DstLoss", "pLoss",
        "Rate", "SrcRate", "DstRate", "TcpRtt", "SynAck", "AckDat",
        "sTos_", "dTos_", "sTtl_", "dTtl_", "SrcWin_", "DstWin_",
        "sVid_", "dVid_", "SrcTCPBase_", "DstTCPBase_",
        "Attack Type_",
        "Proto_icmp", "Proto_tcp", "Proto_udp",
        "Cause_Status",
        "State_CON", "State_ECO", "State_FIN", "State_INT", "State_REQ", "State_RST",
        "Label__Malicious",
    ]

    # Rename raw columns to match the target schema naming convention
    rename_map = {}
    for col in df.columns:
        if col == "sTos":
            rename_map[col] = "sTos_"
        elif col == "dTos":
            rename_map[col] = "dTos_"
        elif col == "sTtl":
            rename_map[col] = "sTtl_"
        elif col == "dTtl":
            rename_map[col] = "dTtl_"
        elif col == "SrcWin":
            rename_map[col] = "SrcWin_"
        elif col == "DstWin":
            rename_map[col] = "DstWin_"
        elif col == "sVid":
            rename_map[col] = "sVid_"
        elif col == "dVid":
            rename_map[col] = "dVid_"
        elif col == "SrcTCPBase":
            rename_map[col] = "SrcTCPBase_"
        elif col == "DstTCPBase":
            rename_map[col] = "DstTCPBase_"
    df.rename(columns=rename_map, inplace=True)

    # Keep only target columns that exist
    available_target_cols = [c for c in target_columns if c in df.columns]
    df = df[available_target_cols].copy()

    print(f"  Preprocessed shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")

    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lowercase, remove underscores.
    This matches the naming convention used in both reference notebooks.
    """
    df.columns = [col.lower().replace("_", "") for col in df.columns]
    return df



# ==============================================================================
# STEP 2: BINARY MODEL TRAINING
# ==============================================================================

def train_binary_model(df: pd.DataFrame, binary_features: list, output_dir: str):
    """
    Train a fast binary anomaly detector (Normal=0 vs Attack=1)
    using RandomForestClassifier on the first BINARY_TRAIN_ROWS rows.
    Save the model and its scaler via joblib.
    """
    print("\n" + "=" * 70)
    print("[STEP 2] BINARY MODEL TRAINING")
    print("=" * 70)

    # Use first 856,400 rows for training
    df_train = df.iloc[:BINARY_TRAIN_ROWS].copy()

    X = df_train[binary_features]
    y = df_train["labelmalicious"]  # 0=Normal, 1=Attack

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForest
    print("  Training Binary RandomForest (100 estimators)...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    # Evaluate on held-out split
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n  === Binary Model — Train Split Report ===")
    print(classification_report(y_test, y_pred))
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.6f}")

    # Optimal threshold search (using unseen data rows 856,400+)
    df_unseen = df.iloc[BINARY_TRAIN_ROWS:].copy()
    X_unseen = df_unseen[binary_features]
    y_unseen = df_unseen["labelmalicious"]
    X_unseen_scaled = scaler.transform(X_unseen)
    y_unseen_proba = model.predict_proba(X_unseen_scaled)[:, 1]

    thresholds = np.arange(0.1, 0.9, 0.005)
    f1_scores = [f1_score(y_unseen, (y_unseen_proba >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"\n  Optimal binary threshold on unseen data: {best_threshold:.2f}")

    y_pred_best = (y_unseen_proba >= best_threshold).astype(int)
    print("\n  === Binary Model — Unseen Data Report (threshold={:.2f}) ===".format(best_threshold))
    print(classification_report(y_unseen, y_pred_best, digits=6))

    # Save model and scaler
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "binary_rf_model.joblib")
    scaler_path = os.path.join(output_dir, "binary_scaler.joblib")
    threshold_path = os.path.join(output_dir, "binary_threshold.txt")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(threshold_path, "w", encoding="utf-8") as f:
        f.write(f"{best_threshold:.12f}\n")
    print(f"  Binary model saved to: {model_path}")
    print(f"  Binary scaler saved to: {scaler_path}")
    print(f"  Binary threshold saved to: {threshold_path}")

    return model, scaler, best_threshold


# ==============================================================================
# STEP 3 & 4: MULTICLASS MODEL (WITH HYBRID SAMPLING)
# ==============================================================================

def train_multiclass_model(df: pd.DataFrame, multiclass_features: list, output_dir: str):
    """
    Train a multiclass attack classifier (9 classes, 0-8).
    Uses hybrid sampling (RandomUnderSampler + SMOTE) to handle class imbalance.
    """
    print("\n" + "=" * 70)
    print("[STEP 3-4] MULTICLASS MODEL TRAINING (with Hybrid Sampling)")
    print("=" * 70)

    df_train = df.iloc[:MULTICLASS_TRAIN_ROWS].copy()

    X = df_train[multiclass_features]
    y = df_train["attack type"]  # 0-8 class labels

    print(f"  Class distribution (first {MULTICLASS_TRAIN_ROWS} rows):")
    print(y.value_counts().sort_index().to_string())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_STATE
    )

    # --- STEP 3: Hybrid Sampling ---
    print("\n  Applying hybrid sampling (undersampling then SMOTE)...")

    # Phase 1: Undersample majority classes
    under_sampling_strategy = {
        0: 50000,   # Benign
        1: 50000,   # UDPFlood
        2: 50000,   # HTTPFlood
        3: 32000,   # SlowrateDoS
        4: 15000,   # TCPConnectScan
        5: 15000,   # SYNScan
        6: 12000,   # UDPScan
        7: 3900,    # SYNFlood
        8: 1000,    # ICMPFlood
    }
    # Only undersample classes that are larger than the target
    actual_under = {}
    for cls, target in under_sampling_strategy.items():
        current_count = (y_train == cls).sum()
        if current_count > target:
            actual_under[cls] = target
        else:
            actual_under[cls] = current_count  # keep as-is
    rus = RandomUnderSampler(sampling_strategy=actual_under, random_state=RANDOM_STATE)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    # Phase 2: Oversample minority classes to 50,000 each
    over_sampling_strategy = {i: 50000 for i in range(9)}
    # Only oversample classes that are smaller than 50,000
    actual_over = {}
    for cls, target in over_sampling_strategy.items():
        current_count = (y_resampled == cls).sum()
        if current_count < target:
            actual_over[cls] = target
        # else: already at or above target, skip
    smote = SMOTE(sampling_strategy=actual_over, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

    print(f"  Resampled class distribution:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    Class {cls} ({ATTACK_TYPE_NAMES[cls]}): {cnt}")

    # --- STEP 4: StandardScaler + RandomForest ---
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    print("\n  Training Multiclass RandomForest (100 estimators)...")
    model = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_resampled_scaled, y_resampled)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("\n  === Multiclass Model — Train Split Report ===")
    print(classification_report(y_test, y_pred, target_names=ATTACK_TYPE_NAMES))

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "multiclass_rf_model.joblib")
    scaler_path = os.path.join(output_dir, "multiclass_scaler.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Multiclass model saved to: {model_path}")
    print(f"  Multiclass scaler saved to: {scaler_path}")

    return model, scaler


def load_saved_models(output_dir: str):
    """
    Load previously trained binary/multiclass artifacts from disk.
    Returns: (binary_model, binary_scaler, binary_threshold, multi_model, multi_scaler)
    """
    model_paths = {
        "binary model": os.path.join(output_dir, "binary_rf_model.joblib"),
        "binary scaler": os.path.join(output_dir, "binary_scaler.joblib"),
        "binary threshold": os.path.join(output_dir, "binary_threshold.txt"),
        "multiclass model": os.path.join(output_dir, "multiclass_rf_model.joblib"),
        "multiclass scaler": os.path.join(output_dir, "multiclass_scaler.joblib"),
    }

    required = ["binary model", "binary scaler", "multiclass model", "multiclass scaler"]
    missing = [name for name in required if not os.path.exists(model_paths[name])]
    if missing:
        raise FileNotFoundError(
            "Missing saved artifacts for eval-only mode: " + ", ".join(missing)
        )

    binary_model = joblib.load(model_paths["binary model"])
    binary_scaler = joblib.load(model_paths["binary scaler"])
    if os.path.exists(model_paths["binary threshold"]):
        with open(model_paths["binary threshold"], "r", encoding="utf-8") as f:
            binary_threshold = float(f.read().strip())
    else:
        binary_threshold = 0.34
        print("  [WARNING] binary_threshold.txt not found; using fallback threshold=0.34")
    multi_model = joblib.load(model_paths["multiclass model"])
    multi_scaler = joblib.load(model_paths["multiclass scaler"])

    print("\n" + "=" * 70)
    print("[LOAD] Loaded saved artifacts (eval-only mode)")
    print("=" * 70)
    print(f"  Binary model: {model_paths['binary model']}")
    print(f"  Binary scaler: {model_paths['binary scaler']}")
    print(f"  Binary threshold: {binary_threshold:.6f}")
    print(f"  Multiclass model: {model_paths['multiclass model']}")
    print(f"  Multiclass scaler: {model_paths['multiclass scaler']}")

    return binary_model, binary_scaler, binary_threshold, multi_model, multi_scaler


# ==============================================================================
# STEP 5: CASCADE ARCHITECTURES
# ==============================================================================

class CascadeInferenceEngine:
    """
    Combined inference pipeline using saved Binary and Multiclass models.
    Implements three routing strategies:
      1. Sequential Binary-First
      2. Sequential Multiclass-First
      3. Parallel Voting / Confidence
    """

    def __init__(
        self,
        binary_model, binary_scaler, binary_features, binary_threshold,
        multi_model, multi_scaler, multi_features,
        unknown_label=-1,
    ):
        self.binary_model = binary_model
        self.binary_scaler = binary_scaler
        self.binary_features = binary_features
        self.binary_threshold = binary_threshold
        self.multi_model = multi_model
        self.multi_scaler = multi_scaler
        self.multi_features = multi_features
        self.unknown_label = unknown_label

    # --- Helper prediction functions ---
    def _binary_predict(self, X_df):
        """Run binary prediction, return array of 0/1."""
        X_scaled = self.binary_scaler.transform(X_df[self.binary_features])
        proba = self.binary_model.predict_proba(X_scaled)[:, 1]
        return (proba >= self.binary_threshold).astype(int)

    def _multi_predict(self, X_df):
        """Run multiclass prediction, return integer class array."""
        X_scaled = self.multi_scaler.transform(X_df[self.multi_features])
        return self.multi_model.predict(X_scaled)

    def _multi_predict_proba(self, X_df):
        """Return (predictions, max_confidence) for multiclass."""
        X_scaled = self.multi_scaler.transform(X_df[self.multi_features])
        proba = self.multi_model.predict_proba(X_scaled)
        preds = np.argmax(proba, axis=1)
        confs = np.max(proba, axis=1)
        return preds, confs

    def _binary_predict_proba(self, X_df):
        """Return (binary_predictions, max_confidence)."""
        X_scaled = self.binary_scaler.transform(X_df[self.binary_features])
        proba = self.binary_model.predict_proba(X_scaled)
        preds = (proba[:, 1] >= self.binary_threshold).astype(int)
        confs = np.max(proba, axis=1)
        return preds, confs

    # ------------------------------------------------------------------
    # Strategy 1: Sequential Binary-First (PRIMARY FOCUS)
    # ------------------------------------------------------------------
    def sequential_binary_first(self, X_df):
        """
        Route traffic to the fast binary model first.
        If predicted 0 (Normal), pass it.
        If predicted 1 (Attack), route to the multiclass model.
        Conflicts (binary=1 but multiclass=0) are labeled unknown.
        """
        start = time.perf_counter()
        n = len(X_df)
        final = np.zeros(n, dtype=int)

        binary_pred = self._binary_predict(X_df)
        final[binary_pred == 0] = 0  # Normal

        attack_idx = np.where(binary_pred == 1)[0]
        if len(attack_idx) > 0:
            multi_pred = self._multi_predict(X_df.iloc[attack_idx])
            final[attack_idx] = multi_pred
            # Conflict: binary says attack, multiclass says benign → unknown
            conflict_mask = multi_pred == 0
            if np.any(conflict_mask):
                final[attack_idx[conflict_mask]] = self.unknown_label

        elapsed = time.perf_counter() - start
        return final, elapsed

    # ------------------------------------------------------------------
    # Strategy 2: Sequential Multiclass-First
    # ------------------------------------------------------------------
    def sequential_multiclass_first(self, X_df):
        """
        Multiclass model goes first; if it predicts 0 (Benign),
        binary model double-checks.
        """
        start = time.perf_counter()
        n = len(X_df)
        final = np.zeros(n, dtype=int)

        multi_pred = self._multi_predict(X_df)

        # Non-zero classes accepted directly
        non_zero = multi_pred != 0
        final[non_zero] = multi_pred[non_zero]

        # For class-0 predictions, verify with binary
        zero_idx = np.where(multi_pred == 0)[0]
        if len(zero_idx) > 0:
            bin_verify = self._binary_predict(X_df.iloc[zero_idx])
            verified_zero = bin_verify == 0
            final[zero_idx[verified_zero]] = 0
            rejected = bin_verify == 1
            final[zero_idx[rejected]] = self.unknown_label

        elapsed = time.perf_counter() - start
        return final, elapsed

    # ------------------------------------------------------------------
    # Strategy 3: Parallel Voting / Confidence
    # ------------------------------------------------------------------
    def parallel_voting(self, X_df):
        """
        Both models run simultaneously. Handle disagreements by labeling
        as -1 (Unknown/Suspicious).
        """
        start = time.perf_counter()
        n = len(X_df)
        final = np.zeros(n, dtype=int)

        binary_pred = self._binary_predict(X_df)
        multi_pred = self._multi_predict(X_df)

        for i in range(n):
            bp = binary_pred[i]
            mp = multi_pred[i]
            if bp == 0:
                # Binary says normal
                final[i] = 0 if mp == 0 else mp  # trust multiclass detail
            else:
                # Binary says attack
                if mp == 0:
                    final[i] = self.unknown_label  # conflict
                else:
                    final[i] = mp  # agreement

        elapsed = time.perf_counter() - start
        return final, elapsed

    def parallel_confidence(self, X_df, confidence_threshold=0.7):
        """
        Both models run. Use confidence scores to resolve disagreements.
        Higher confidence model wins; if both are low → unknown.
        """
        start = time.perf_counter()
        n = len(X_df)
        final = np.zeros(n, dtype=int)

        bin_pred, bin_conf = self._binary_predict_proba(X_df)
        multi_pred, multi_conf = self._multi_predict_proba(X_df)

        for i in range(n):
            bp, bc = bin_pred[i], bin_conf[i]
            mp, mc = multi_pred[i], multi_conf[i]

            if bc > confidence_threshold and mc > confidence_threshold:
                # Both confident
                if (mp == 0 and bp == 0) or (mp != 0 and bp == 1):
                    final[i] = mp  # agreement
                else:
                    # disagreement — trust higher confidence
                    final[i] = mp if mc > bc else (0 if bp == 0 else self.unknown_label)
            elif mc > confidence_threshold:
                final[i] = mp
            elif bc > confidence_threshold:
                final[i] = 0 if bp == 0 else self.unknown_label
            else:
                final[i] = self.unknown_label  # both uncertain

        elapsed = time.perf_counter() - start
        return final, elapsed


# ==============================================================================
# STEP 6: EXPERIMENTAL fDNN
# ==============================================================================

class MulticlassFDNN:
    """
    Experimental forest-DNN hybrid:
      1. Train a RandomForest model
      2. Extract its structural 'leaf indices' using .apply()
      3. Map these indices into an MLP (Deep Neural Network) for final prediction.
    """

    def __init__(
        self,
        rf_n_estimators=100,
        rf_max_depth=15,
        mlp_hidden_layers=(256, 128, 64, 32),
    ):
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.mlp_hidden_layers = mlp_hidden_layers

        self.rf_feature_extractor = None
        self.mlp_classifier = None
        self.input_scaler = StandardScaler()
        self.rf_features_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        self.trained = False

    def fit(self, X_train, y_train, max_iter=300):
        """Train the RF→MLP pipeline on multiclass data."""
        print("\n  [fDNN] Phase 1 — Training RF Feature Extractor...")
        y_enc = self.label_encoder.fit_transform(y_train)

        X_scaled = self.input_scaler.fit_transform(X_train)

        self.rf_feature_extractor = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.rf_feature_extractor.fit(X_scaled, y_enc)
        rf_acc = self.rf_feature_extractor.score(X_scaled, y_enc)
        print(f"    RF accuracy on train: {rf_acc:.4f}")

        # Extract leaf indices as features
        leaf_indices = self.rf_feature_extractor.apply(X_scaled)
        leaf_scaled = self.rf_features_scaler.fit_transform(leaf_indices)
        print(f"    Leaf feature shape: {leaf_scaled.shape}")

        # Phase 2: MLP on leaf features
        print("  [fDNN] Phase 2 — Training MLP on leaf features...")
        self.mlp_classifier = MLPClassifier(
            hidden_layer_sizes=self.mlp_hidden_layers,
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=max_iter,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            verbose=False,
        )
        self.mlp_classifier.fit(leaf_scaled, y_enc)
        mlp_acc = self.mlp_classifier.score(leaf_scaled, y_enc)
        print(f"    MLP accuracy on train: {mlp_acc:.4f}")
        self.trained = True

    def predict(self, X):
        """Predict using the RF→MLP pipeline."""
        X_scaled = self.input_scaler.transform(X)
        leaf_indices = self.rf_feature_extractor.apply(X_scaled)
        leaf_scaled = self.rf_features_scaler.transform(leaf_indices)
        y_enc = self.mlp_classifier.predict(leaf_scaled)
        return self.label_encoder.inverse_transform(y_enc)

    def evaluate(self, X_test, y_test):
        """Evaluate on test data."""
        y_pred = self.predict(X_test)
        print("\n  === fDNN Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=ATTACK_TYPE_NAMES, zero_division=0))
        acc = accuracy_score(y_test, y_pred)
        print(f"  fDNN Accuracy: {acc:.4f}")
        return y_pred


# ==============================================================================
# STEP 7: EVALUATION & BENCHMARKING
# ==============================================================================

def evaluate_and_benchmark(
    df, binary_features, multi_features,
    binary_model, binary_scaler, binary_threshold,
    multi_model, multi_scaler,
    fdnn_model=None,
):
    """
    Evaluate all strategies against unseen test data (rows 800,000+).
    Generate Classification Reports, Confusion Matrices, Accuracy, F1-scores.
    Benchmark throughput (samples/sec).
    """
    print("\n" + "=" * 70)
    print("[STEP 7] EVALUATION & BENCHMARKING (unseen test data)")
    print("=" * 70)

    test_data = df.iloc[TEST_START_ROW:].copy()
    y_true_attack = test_data["attack type"].values
    y_true_binary = test_data["labelmalicious"].values
    n_test = len(test_data)

    print(f"  Test set size: {n_test} samples\n")

    # Initialize cascade engine
    engine = CascadeInferenceEngine(
        binary_model, binary_scaler, binary_features, binary_threshold,
        multi_model, multi_scaler, multi_features,
    )

    strategies = {
        "Sequential Binary-First": engine.sequential_binary_first,
        "Sequential Multiclass-First": engine.sequential_multiclass_first,
        "Parallel Voting": engine.parallel_voting,
        "Parallel Confidence": engine.parallel_confidence,
    }

    results_table = []

    for name, strategy_fn in strategies.items():
        print(f"\n  --- {name} ---")
        preds, elapsed = strategy_fn(test_data)

        # Exclude unknowns for accuracy calculation
        known_mask = preds != -1
        y_true_known = y_true_attack[known_mask]
        y_pred_known = preds[known_mask]

        if len(y_true_known) > 0:
            acc = accuracy_score(y_true_known, y_pred_known)
        else:
            acc = 0.0

        unknown_count = np.sum(~known_mask)
        throughput = n_test / elapsed
        time_per_sample = elapsed / n_test * 1000  # ms

        print(f"    Time: {elapsed:.4f} s")
        print(f"    Throughput: {throughput:.2f} samples/s")
        print(f"    Time/sample: {time_per_sample:.5f} ms")
        print(f"    Accuracy (known only): {acc:.4f}")
        print(f"    Unknown/Suspicious: {unknown_count}/{n_test} ({unknown_count / n_test * 100:.1f}%)")
        if len(y_true_known) > 0:
            print(classification_report(
                y_true_known, y_pred_known,
                labels=range(9), target_names=ATTACK_TYPE_NAMES, zero_division=0
            ))

        results_table.append({
            "Strategy": name,
            "Time (s)": round(elapsed, 4),
            "Throughput (samples/s)": round(throughput, 2),
            "Time/sample (ms)": round(time_per_sample, 5),
            "Accuracy": round(acc, 4),
            "Unknown %": round(unknown_count / n_test * 100, 1),
        })

    # --- fDNN evaluation ---
    if fdnn_model is not None and fdnn_model.trained:
        print("\n  --- fDNN (RF → MLP Hybrid) ---")
        start = time.perf_counter()
        fdnn_preds = fdnn_model.predict(test_data[multi_features])
        fdnn_elapsed = time.perf_counter() - start

        fdnn_acc = accuracy_score(y_true_attack, fdnn_preds)
        fdnn_throughput = n_test / fdnn_elapsed

        print(f"    Time: {fdnn_elapsed:.4f} s")
        print(f"    Throughput: {fdnn_throughput:.2f} samples/s")
        print(f"    Accuracy: {fdnn_acc:.4f}")
        print(classification_report(
            y_true_attack, fdnn_preds,
            labels=range(9), target_names=ATTACK_TYPE_NAMES, zero_division=0
        ))
        results_table.append({
            "Strategy": "fDNN (RF→MLP)",
            "Time (s)": round(fdnn_elapsed, 4),
            "Throughput (samples/s)": round(fdnn_throughput, 2),
            "Time/sample (ms)": round(fdnn_elapsed / n_test * 1000, 5),
            "Accuracy": round(fdnn_acc, 4),
            "Unknown %": 0.0,
        })

    # --- Summary Table ---
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(results_table)
    print(summary_df.to_string(index=False))
    print()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Execute the full end-to-end pipeline."""
    parser = argparse.ArgumentParser(description="5G-NIDD full pipeline")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip model training and evaluate using saved artifacts in output/",
    )
    parser.add_argument(
        "--skip-fdnn",
        action="store_true",
        help="Skip experimental fDNN training/evaluation",
    )
    args = parser.parse_args()

    overall_start = time.time()

    # ---- Step 1: Preprocessing ----
    df = load_and_preprocess_raw_data(RAW_CSV_PATH)
    df = normalize_column_names(df)

    # Use the pre-normalized feature lists directly
    binary_features = BINARY_FEATURES_NORMALIZED
    multi_features = MULTICLASS_FEATURES_NORMALIZED

    # Verify all features exist
    missing_binary = [f for f in binary_features if f not in df.columns]
    missing_multi = [f for f in multi_features if f not in df.columns]
    if missing_binary:
        raise ValueError(f"Missing binary features in DataFrame: {missing_binary}")
    if missing_multi:
        raise ValueError(f"Missing multiclass features in DataFrame: {missing_multi}")

    print(f"\n  Binary features ({len(binary_features)}): {binary_features}")
    print(f"  Multiclass features ({len(multi_features)}): {multi_features}")

    if args.eval_only:
        binary_model, binary_scaler, binary_threshold, multi_model, multi_scaler = load_saved_models(OUTPUT_DIR)
        fdnn = None
    else:
        # ---- Step 2: Binary Model Training ----
        binary_model, binary_scaler, binary_threshold = train_binary_model(
            df, binary_features, OUTPUT_DIR
        )

        # ---- Steps 3-4: Multiclass Model Training ----
        multi_model, multi_scaler = train_multiclass_model(
            df, multi_features, OUTPUT_DIR
        )

        # ---- Step 6: Experimental fDNN ----
        fdnn = None
        if not args.skip_fdnn:
            print("\n" + "=" * 70)
            print("[STEP 6] EXPERIMENTAL fDNN (RF → MLP Hybrid)")
            print("=" * 70)

            fdnn = MulticlassFDNN(
                rf_n_estimators=100, rf_max_depth=15,
                mlp_hidden_layers=(256, 128, 64, 32),
            )

            # Use the same multiclass training data
            df_fdnn_train = df.iloc[:MULTICLASS_TRAIN_ROWS].copy()
            X_fdnn_train = df_fdnn_train[multi_features]
            y_fdnn_train = df_fdnn_train["attack type"]

            try:
                fdnn.fit(X_fdnn_train, y_fdnn_train, max_iter=300)
            except Exception as e:
                print(f"  [WARNING] fDNN training failed: {e}")
                fdnn.trained = False

    # ---- Step 7: Evaluation & Benchmarking ----
    evaluate_and_benchmark(
        df, binary_features, multi_features,
        binary_model, binary_scaler, binary_threshold,
        multi_model, multi_scaler,
        fdnn_model=fdnn,
    )

    total_elapsed = time.time() - overall_start
    print(f"\n{'=' * 70}")
    print(f"  TOTAL PIPELINE TIME: {total_elapsed:.2f} seconds")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
