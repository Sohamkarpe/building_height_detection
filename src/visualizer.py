"""
Results Visualizer
Author: Soham Ashok Karpe, M. Eng.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class ResultVisualizer:
    """Visualizes height estimation results on images and charts."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, image: np.ndarray, height_m: float,
             features: np.ndarray = None, title: str = "") -> np.ndarray:
        """Draw estimated height on image."""
        output = image.copy()
        h, w = output.shape[:2]

        # Dark overlay at bottom
        overlay = output.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

        # Height display
        height_text = f"Estimated Height: {height_m:.1f} m"
        cv2.putText(output, height_text, (15, h - 45),
                    self.font, 0.8, (0, 212, 255), 2)

        # Floors estimate
        floors = max(1, int(height_m / 3.2))
        floors_text = f"~{floors} floors"
        cv2.putText(output, floors_text, (15, h - 15),
                    self.font, 0.6, (180, 180, 180), 1)

        # Title
        if title:
            cv2.putText(output, title, (15, 30),
                        self.font, 0.6, (255, 255, 255), 1)

        return output

    def plot_results(self, results: list, save_path: str = None):
        """Plot bar chart of height results."""
        names = [Path(r["image"]).stem[:20] for r in results]
        heights = [r["height_m"] for r in results]

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
        bars = ax.bar(names, heights, color="#00d4ff", edgecolor="#0099bb", linewidth=1.2)

        ax.set_title("Building Height Estimation Results", fontsize=14, fontweight="bold")
        ax.set_xlabel("Building")
        ax.set_ylabel("Estimated Height (m)")
        ax.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, val in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}m", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_feature_importance(self, feature_names: list, importances: np.ndarray):
        """Plot feature importance (for Random Forest)."""
        idx = np.argsort(importances)[::-1][:15]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(idx)), importances[idx], color="#7b2dff")
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
        ax.set_title("Feature Importance for Height Estimation")
        ax.set_ylabel("Importance")
        plt.tight_layout()
        plt.show()
