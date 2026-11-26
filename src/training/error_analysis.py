
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


# ==============================
# Load Predictions
# ==============================

DATA_PATH = r"D:\projects\Depi_CV\fight_detection_system\outputs\evaluation\predictions.npz"
OUTPUT_DIR = r"D:\projects\Depi_CV\fight_detection_system\outputs\error_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = np.load(DATA_PATH)

y_true = data["y_true"]
y_prob = data["y_prob"]


# ==============================
# Threshold Optimization
# ==============================

thresholds = np.arange(0.1, 0.91, 0.05)

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    })


# ==============================
# Find Best Threshold
# ==============================

best_result = max(
    results,
    key=lambda x: x["f1"]
)

print("\nBest Threshold Results")

for key, value in best_result.items():
    print(f"{key}: {value:.4f}")


# ==============================
# Plot Threshold Comparison
# ==============================

plt.figure(figsize=(10, 6))

plt.plot(
    thresholds,
    [r["precision"] for r in results],
    marker="o",
    label="Precision"
)

plt.plot(
    thresholds,
    [r["recall"] for r in results],
    marker="o",
    label="Recall"
)

plt.plot(
    thresholds,
    [r["f1"] for r in results],
    marker="o",
    label="F1 Score"
)

plt.plot(
    thresholds,
    [r["accuracy"] for r in results],
    marker="o",
    label="Accuracy"
)

plt.axvline(
    best_result["threshold"],
    linestyle="--",
    label="Best Threshold"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization")

plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "threshold_analysis.png"),
    bbox_inches="tight"
)

plt.close()


# ==============================
# Error Analysis
# ==============================

best_threshold = best_result["threshold"]

y_pred = (y_prob >= best_threshold).astype(int)

cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\nError Analysis")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)


# ==============================
# Identify Wrong Predictions
# ==============================

wrong_indices = np.where(y_true != y_pred)[0]

print("\nIncorrect Predictions:", len(wrong_indices))
print("Wrong Prediction Indices:", wrong_indices)


# ==============================
# Save Best Threshold
# ==============================

with open(
    os.path.join(OUTPUT_DIR, "best_threshold.txt"),
    "w"
) as file:

    file.write(str(best_threshold))


print("\nError analysis completed!")
