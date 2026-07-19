"""
Building Height Detection using AI
Author: Soham Ashok Karpe, M. Eng.
Institution: Technische Hochschule Deggendorf (DIT), Campus Cham
Period: Mar 2023 – Jul 2023 | DIT HiWi Project
"""

import argparse
import sys
import os
import cv2
import yaml
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from feature_extractor import FeatureExtractor
from height_estimator import HeightEstimator
from visualizer import ResultVisualizer
from batch_processor import BatchProcessor
from utils.logger import setup_logger
from utils.data_export import DataExporter

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Building Height Detection using AI"
    )
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument(
        "--model", type=str, default="random_forest",
        choices=["random_forest", "svr", "neural_network"],
        help="ML model to use"
    )
    parser.add_argument(
        "--config", type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        choices=["csv", "json"],
        help="Export results format"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show visualization"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/",
        help="Output directory"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config not found: {path}, using defaults")
        return {}


def estimate_single(image_path: str, extractor, estimator, visualizer, args, config):
    """Estimate height for a single image."""
    logger.info(f"Processing: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Cannot read image: {image_path}")
        return None

    # Extract features
    features = extractor.extract(img)
    logger.info(f"Features extracted: {len(features)} dimensions")

    # Estimate height
    height_m = estimator.predict(features)
    logger.info(f"Estimated height: {height_m:.2f} m")

    # Visualize
    if args.visualize:
        vis = visualizer.draw(img, height_m, features, Path(image_path).name)
        cv2.imshow("Building Height Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {"image": image_path, "height_m": round(height_m, 2)}


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("Building Height Detection — Starting")
    logger.info("Author: Soham Karpe, M. Eng. | DIT Campus Cham")
    logger.info("=" * 60)

    config = load_config(args.config)

    # Initialize components
    extractor = FeatureExtractor(config)
    estimator = HeightEstimator(model_type=args.model, config=config)
    visualizer = ResultVisualizer(config)
    exporter = DataExporter(output_dir=args.output)

    os.makedirs(args.output, exist_ok=True)

    results = []

    if args.image:
        result = estimate_single(args.image, extractor, estimator, visualizer, args, config)
        if result:
            results.append(result)
            print(f"\n📐 Estimated Height: {result['height_m']} meters")

    elif args.folder:
        processor = BatchProcessor(extractor, estimator, visualizer, config)
        results = processor.process_folder(args.folder)
        print(f"\n✅ Processed {len(results)} images")
        for r in results:
            print(f"  {Path(r['image']).name}: {r['height_m']} m")
    else:
        logger.error("Please provide --image or --folder")
        sys.exit(1)

    # Export results
    if args.export and results:
        out_path = exporter.export(results, format=args.export)
        logger.info(f"Results exported to: {out_path}")


if __name__ == "__main__":
    main()
