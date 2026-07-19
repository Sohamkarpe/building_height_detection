"""
Height Estimator - ML Models
Author: Soham Ashok Karpe, M. Eng.
"""

import numpy as np
import os
import joblib
from utils.logger import setup_logger

logger = setup_logger(__name__)


class HeightEstimator:
    """
    Estimates building height in meters using ML regression models.

    Supported models:
    - Random Forest Regressor
    - Support Vector Regression (SVR)
    - Neural Network (MLP Regressor)
    """

    MODELS = ["random_forest", "svr", "neural_network"]

    def __init__(self, model_type: str = "random_forest", config: dict = None):
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.scaler = None
        self._load_or_init_model()

    def _load_or_init_model(self):
        """Load saved model or initialize a new untrained one."""
        model_dir = self.config.get("model", {}).get("save_dir", "models/")
        model_path = os.path.join(model_dir, f"{self.model_type}.pkl")

        if os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
        else:
            logger.warning(
                f"No saved model found at {model_path}. "
                "Initializing untrained model. Run train.py first."
            )
            self.model = self._build_model()

    def _build_model(self):
        """Build the selected ML model."""
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )

        elif self.model_type == "svr":
            from sklearn.svm import SVR
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            return Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=10, gamma="auto", epsilon=0.5))
            ])

        elif self.model_type == "neural_network":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model on features X and target heights y."""
        from sklearn.preprocessing import StandardScaler

        logger.info(f"Training {self.model_type} on {len(X)} samples...")

        if self.model_type != "svr":
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        self.model.fit(X_scaled, y)
        logger.info("Training complete!")

    def predict(self, features: np.ndarray) -> float:
        """
        Predict building height from feature vector.

        Args:
            features: 1D feature vector

        Returns:
            Estimated height in meters
        """
        if self.model is None:
            logger.warning("Model not trained. Returning default estimate.")
            return 15.0  # default placeholder

        X = features.reshape(1, -1)

        if self.scaler is not None and self.model_type != "svr":
            X = self.scaler.transform(X)

        try:
            height = float(self.model.predict(X)[0])
            # Clamp to reasonable range (2m to 500m)
            height = max(2.0, min(500.0, height))
            return height
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 15.0

    def save(self, save_dir: str = "models/"):
        """Save model to disk."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.model_type}.pkl")
        joblib.dump(self.model, path)
        if self.scaler:
            joblib.dump(self.scaler, os.path.join(save_dir, "scaler.pkl"))
        logger.info(f"Model saved to: {path}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        X_eval = X
        if self.scaler and self.model_type != "svr":
            X_eval = self.scaler.transform(X)

        y_pred = self.model.predict(X_eval)

        return {
            "MAE": round(mean_absolute_error(y, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y, y_pred)), 3),
            "R2": round(r2_score(y, y_pred), 3),
        }
