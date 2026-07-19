"""
Image Feature Extractor for Building Height Estimation
Author: Soham Ashok Karpe, M. Eng.
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureExtractor:
    """
    Extracts visual features from building images for height estimation.

    Features:
    - Edge density (Canny)
    - Shadow analysis
    - HOG (Histogram of Oriented Gradients)
    - LBP (Local Binary Patterns) texture
    - Contour / shape features
    - Color histogram
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.target_size = tuple(
            self.config.get("features", {}).get("target_size", [256, 256])
        )

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all features from an image.

        Args:
            image: BGR image (numpy array)

        Returns:
            Feature vector (1D numpy array)
        """
        img = cv2.resize(image, self.target_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = np.concatenate([
            self._edge_features(gray),
            self._hog_features(gray),
            self._texture_features(gray),
            self._contour_features(gray),
            self._color_features(img),
            self._shadow_features(gray),
        ])

        logger.debug(f"Feature vector shape: {features.shape}")
        return features

    def _edge_features(self, gray: np.ndarray) -> np.ndarray:
        """Canny edge detection features."""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return np.array([
            edge_density,
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude),
        ])

    def _hog_features(self, gray: np.ndarray) -> np.ndarray:
        """Histogram of Oriented Gradients features."""
        features, _ = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=True
        )
        # Reduce to summary stats to keep feature vector manageable
        return np.array([
            np.mean(features),
            np.std(features),
            np.max(features),
            np.median(features),
        ])

    def _texture_features(self, gray: np.ndarray) -> np.ndarray:
        """Local Binary Pattern texture features."""
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=26, range=(0, 26), density=True)
        # Return summary stats
        return np.array([np.mean(hist), np.std(hist), np.max(hist)])

    def _contour_features(self, gray: np.ndarray) -> np.ndarray:
        """Building contour and shape features."""
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros(5)

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = h / w if w > 0 else 0
        solidity = area / (w * h) if (w * h) > 0 else 0

        return np.array([
            area / (gray.shape[0] * gray.shape[1]),
            aspect_ratio,
            solidity,
            h / gray.shape[0],
            w / gray.shape[1],
        ])

    def _color_features(self, img: np.ndarray) -> np.ndarray:
        """Color histogram features (HSV)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        # Normalize
        h_hist = h_hist / (h_hist.sum() + 1e-7)
        s_hist = s_hist / (s_hist.sum() + 1e-7)
        return np.concatenate([h_hist, s_hist])

    def _shadow_features(self, gray: np.ndarray) -> np.ndarray:
        """
        Shadow analysis for height estimation.
        Longer shadows typically indicate taller buildings.
        """
        # Threshold dark regions as shadows
        _, shadow_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        shadow_ratio = np.sum(shadow_mask > 0) / shadow_mask.size

        # Shadow bounding box
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            _, _, sw, sh = cv2.boundingRect(largest)
            shadow_aspect = sh / sw if sw > 0 else 0
        else:
            shadow_aspect = 0

        return np.array([shadow_ratio, shadow_aspect])
