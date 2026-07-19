"""
Batch Image Processor
Author: Soham Ashok Karpe, M. Eng.
"""

import cv2
import os
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class BatchProcessor:
    """Process a folder of building images in batch."""

    def __init__(self, extractor, estimator, visualizer, config: dict = None):
        self.extractor = extractor
        self.estimator = estimator
        self.visualizer = visualizer
        self.config = config or {}

    def process_folder(self, folder_path: str) -> list:
        """Process all images in a folder."""
        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"Folder not found: {folder_path}")
            return []

        image_files = [
            f for f in folder.iterdir()
            if f.suffix.lower() in SUPPORTED_FORMATS
        ]

        if not image_files:
            logger.warning(f"No images found in: {folder_path}")
            return []

        logger.info(f"Processing {len(image_files)} images from: {folder_path}")
        results = []

        for img_path in sorted(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Cannot read: {img_path.name}")
                continue

            features = self.extractor.extract(img)
            height_m = self.estimator.predict(features)

            results.append({
                "image": str(img_path),
                "filename": img_path.name,
                "height_m": round(height_m, 2),
                "floors_estimate": max(1, int(height_m / 3.2)),
            })
            logger.info(f"  {img_path.name}: {height_m:.2f} m")

        logger.info(f"Batch complete: {len(results)}/{len(image_files)} processed")
        return results
