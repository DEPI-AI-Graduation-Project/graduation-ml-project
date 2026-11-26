
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

from src.preprocessing.dataset import create_dataset


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = r"D:\projects\Depi_CV\fight_detection_system\models\checkpoints\best_phase2.keras"

VAL_DIR = r"D:\projects\Depi_CV\fight_detection_system\data\raw\RWF-2000\val"

OUTPUT_DIR = r"D:\projects\Depi_CV\fight_detection_system\outputs\evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# LOAD VIDEO PATHS AND LABELS
# ==========================================

def collect_video_paths(data_dir):

    video_paths = []
    labels = []

    classes = {
        "NonFight": 0,
        "Fight": 1
    }

    for class_name, label in classes.items():

        class_dir = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Directory not found: {class_dir}"
            )

        for filename in sorted(os.listdir(class_dir)):

            if filename.lower().endswith(
                (".avi", ".mp4", ".mov", ".mkv")
            ):

                video_paths.append(
                    os.path.join(class_dir, filename)
                )

                labels.append(label)

    return video_paths, np.array(labels)


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

model.summary()


# ==========================================
# LOAD VALIDATION DATASET
# ==========================================

print("\nLoading validation dataset...")

val_dataset = create_dataset(
    VAL_DIR,
    training=False
)

video_paths, original_labels = collect_video_paths(
    VAL_DIR
)

print("Validation videos:", len(video_paths))


# ==========================================
# COLLECT TRUE LABELS
# ==========================================

y_true = []

for videos, labels in val_dataset:

    y_true.extend(
        labels.numpy().reshape(-1)
    )

y_true = np.array(y_true).astype(int)

print("True labels collected:", len(y_true))


# ==========================================
# ALIGNMENT CHECK
# ==========================================

if len(video_paths) != len(y_true):

    raise ValueError(
        "Mismatch between video paths and dataset labels!"
    )

if not np.array_equal(original_labels, y_true):

    raise ValueError(
        "Video paths and dataset labels are not aligned! "
        "Check class ordering and dataset shuffling."
    )


# ==========================================
# MODEL PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

y_prob = model.predict(
    val_dataset,
    verbose=1
).reshape(-1)

y_pred = (y_prob >= 0.5).astype(int)


# ==========================================
# ALIGNMENT CHECK AGAIN
# ==========================================

if not (
    len(y_true) == len(y_prob) == len(video_paths)
):

    raise ValueError("Prediction count mismatch!")


# ==========================================
# CLASSIFICATION METRICS
# ==========================================

print("\n========== EVALUATION RESULTS ==========")

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["NonFight", "Fight"],
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["NonFight", "Fight"],
    yticklabels=["NonFight", "Fight"]
)

plt.title("Fight Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    bbox_inches="tight"
)

plt.close()

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("Fight Detection - ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve.png"),
    bbox_inches="tight"
)

plt.close()

print("ROC AUC:", round(roc_auc, 4))


# ==========================================
# VIDEO PREDICTION RESULTS
# ==========================================

results = pd.DataFrame({

    "Video": [
        os.path.basename(path)
        for path in video_paths
    ],

    "Path": video_paths,

    "Actual": y_true,

    "Probability": y_prob,

    "Predicted": y_pred

})


# ==========================================
# ERROR CLASSIFICATION
# ==========================================

results["Actual_Class"] = results["Actual"].map({
    0: "NonFight",
    1: "Fight"
})

results["Predicted_Class"] = results["Predicted"].map({
    0: "NonFight",
    1: "Fight"
})

results["Error_Type"] = np.select(

    [
        (results["Actual"] == 0) &
        (results["Predicted"] == 1),

        (results["Actual"] == 1) &
        (results["Predicted"] == 0)
    ],

    [
        "False Positive",
        "False Negative"
    ],

    default="Correct"

)


# ==========================================
# SAVE ALL PREDICTIONS
# ==========================================

results.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "video_predictions.csv"
    ),

    index=False

)


# ==========================================
# EXTRACT MISCLASSIFIED VIDEOS
# ==========================================

errors = results[
    results["Error_Type"] != "Correct"
]

false_positives = results[
    results["Error_Type"] == "False Positive"
]

false_negatives = results[
    results["Error_Type"] == "False Negative"
]


errors.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "misclassified_videos.csv"
    ),

    index=False

)


# Save each error type separately

false_positives.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "false_positives.csv"
    ),
    index=False
)

false_negatives.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "false_negatives.csv"
    ),
    index=False
)


# ==========================================
# SAVE NUMERICAL RESULTS
# ==========================================

np.savez(

    os.path.join(
        OUTPUT_DIR,
        "predictions.npz"
    ),

    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob

)


# ==========================================
# PRINT ERROR ANALYSIS
# ==========================================

print("\n========== ERROR ANALYSIS ==========")

print("Total videos:", len(results))

print("Correct predictions:", len(results) - len(errors))

print("Misclassified videos:", len(errors))

print("False Positives:", len(false_positives))

print("False Negatives:", len(false_negatives))


print("\nFirst 10 Misclassified Videos:\n")

print(

    errors[[
        "Video",
        "Actual_Class",
        "Probability",
        "Predicted_Class",
        "Error_Type"
    ]].head(10).to_string(index=False)

)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========== SAVED FILES ==========")

print("confusion_matrix.png")
print("roc_curve.png")
print("predictions.npz")
print("video_predictions.csv")
print("misclassified_videos.csv")
print("false_positives.csv")
print("false_negatives.csv")

print("\nEvaluation completed successfully!")
