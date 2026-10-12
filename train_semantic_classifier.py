import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ServiceSense.csv"

EMBEDDINGS_PATH = (
    BASE_DIR
    / "models"
    / "advanced_dense_embeddings.npy"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "semantic_intent_classifier.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SERVICESENSE AI - SEMANTIC INTENT CLASSIFIER")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df = df[
    ["instruction", "category", "intent", "response"]
].copy()

df = df.dropna(
    subset=["instruction", "intent"]
).reset_index(drop=True)

print("Dataset rows:", len(df))


# ============================================================
# LOAD EXISTING MINILM EMBEDDINGS
# ============================================================

print("\nLoading saved MiniLM embeddings...")

X = np.load(
    EMBEDDINGS_PATH
).astype("float32")

y = df["intent"].astype(str).to_numpy()

print("Embedding shape:", X.shape)
print("Number of labels:", len(y))


# ============================================================
# SAFETY CHECK
# ============================================================

if len(X) != len(y):

    raise ValueError(
        "Embedding count does not match dataset rows. "
        "Do not train because row alignment is uncertain."
    )

print("Embedding / dataset alignment count: OK")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# SEMANTIC CLASSIFIER
# ============================================================

print("\nTraining Semantic Logistic Regression...")
print("Features: MiniLM 384-dimensional embeddings")
print("This may take a little time on CPU.")

model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    
)

model.fit(
    X_train,
    y_train
)

print("Training completed")


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating semantic classifier...")

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )
)

print("\n" + "=" * 70)
print("SEMANTIC CLASSIFIER TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

print(
    "\nSemantic classifier saved:"
)

print(MODEL_PATH)

print("\nNumber of intent classes:")
print(len(model.classes_))

print("\nIntent classes:")
print(list(model.classes_))

print("\n" + "=" * 70)
print("SEMANTIC CLASSIFIER READY")
print("=" * 70)