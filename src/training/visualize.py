
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_training_history(history1, history2, output_dir="outputs"):

    os.makedirs(output_dir, exist_ok=True)

    metrics = ["loss", "accuracy", "precision", "recall", "auc"]

    for metric in metrics:

        plt.figure(figsize=(10, 5))

        # Phase 1
        plt.plot(
            history1.history[metric],
            label=f"Phase 1 Train {metric}"
        )

        plt.plot(
            history1.history[f"val_{metric}"],
            label=f"Phase 1 Val {metric}"
        )

        # Phase 2
        offset = len(history1.history[metric])

        phase2_epochs = range(
            offset,
            offset + len(history2.history[metric])
        )

        plt.plot(
            phase2_epochs,
            history2.history[metric],
            label=f"Phase 2 Train {metric}"
        )

        plt.plot(
            phase2_epochs,
            history2.history[f"val_{metric}"],
            label=f"Phase 2 Val {metric}"
        )

        plt.title(f"Training and Validation {metric.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())

        plt.legend()
        plt.grid(True)

        plt.savefig(
            os.path.join(output_dir, f"{metric}_curve.png")
        )

        plt.close()

    print("Training curves saved successfully!")
