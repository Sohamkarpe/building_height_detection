"""Export results to CSV or JSON."""
import os
import json
import csv
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataExporter:
    def __init__(self, output_dir: str = "results/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export(self, results: list, format: str = "csv") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if format == "csv":
            return self._to_csv(results, ts)
        elif format == "json":
            return self._to_json(results, ts)

    def _to_csv(self, results: list, ts: str) -> str:
        path = os.path.join(self.output_dir, f"results_{ts}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"CSV saved: {path}")
        return path

    def _to_json(self, results: list, ts: str) -> str:
        path = os.path.join(self.output_dir, f"results_{ts}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"JSON saved: {path}")
        return path
