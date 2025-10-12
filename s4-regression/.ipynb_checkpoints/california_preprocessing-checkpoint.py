
"""
Comprehensive preprocessing for the California Housing Dataset
=============================================================

This module prepares the California Housing data for regression models.
It provides:
- Data loading
- Feature diagnostics
- Robust preprocessing pipeline (impute → outlier clip → feature engineering → optional power transform → scaling)
- Target transformation (log1p) + inverse utilities
- Train/validation split helper
- Persist/load utilities for the fitted preprocessor
- Convenience CLI: run `python california_preprocessing_complete.py` to materialize processed arrays

Compared to a minimal pipeline (impute+scale), this adds feature engineering and
defensive steps while keeping defaults sensible for linear models.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class PreprocessConfig:
    test_size: float = 0.2
    random_state: int = 42

    # Outlier clipping (quantile-based). Set to None to disable.
    clip_lower_q: Optional[float] = 0.005
    clip_upper_q: Optional[float] = 0.995

    # Scaling: "standard" | "robust" | None
    scaler: Optional[str] = "standard"

    # Nonlinear transforms: "power" | "quantile" | None
    nonlin: Optional[str] = None  # for linear models try "power" sometimes

    # Target transform
    log1p_target: bool = True

# -----------------------------
# Utilities
# -----------------------------

def _diagnostics(df: pd.DataFrame, name: str = "X") -> pd.DataFrame:
    """Return a diagnostic table for basic data health checks."""
    desc = df.describe().T
    miss = df.isna().mean()
    skew = df.skew(numeric_only=True)
    out = desc[['mean', 'std', 'min', 'max']].copy()
    out['missing_frac'] = miss
    out['skew'] = skew
    out.index.name = f"{name}_feature"
    return out

class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features by per-column quantiles to tame extreme outliers."""
    def __init__(self, lower_q: float = 0.005, upper_q: float = 0.995):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = self._as_df(X)
        self.columns_ = X.columns
        if self.lower_q is None or self.upper_q is None:
            self.lower_ = None
            self.upper_ = None
            return self
        qs = X.quantile([self.lower_q, self.upper_q])
        self.lower_ = qs.iloc[0].values
        self.upper_ = qs.iloc[1].values
        return self

    def transform(self, X):
        X = self._as_df(X)
        if self.lower_ is None or self.upper_ is None:
            return X.values
        Xc = X.copy()
        for i, col in enumerate(self.columns_):
            Xc[col] = np.clip(Xc[col].values, self.lower_[i], self.upper_[i])
        return Xc.values

    @staticmethod
    def _as_df(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Add domain-reasonable engineered features for California Housing.
    Original features (all numeric): 
    ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
    Engineered features:
      - bedrooms_per_room = AveBedrms / AveRooms
      - rooms_per_person = AveRooms / AveOccup
      - pop_per_occup = Population / AveOccup
      - latlong_interact = Latitude * Longitude
    """
    def __init__(self, passthrough=True):
        self.passthrough = passthrough
        self.original_columns_ = None
        self.columns_out_ = None

    def fit(self, X, y=None):
        X = self._as_df(X)
        self.original_columns_ = list(X.columns)
        self.columns_out_ = self.original_columns_ + [
            "bedrooms_per_room",
            "rooms_per_person",
            "pop_per_occup",
            "latlong_interact",
        ]
        return self

    def transform(self, X):
        X = self._as_df(X).copy()
        eps = 1e-9
        X["bedrooms_per_room"] = X["AveBedrms"] / (X["AveRooms"] + eps)
        X["rooms_per_person"] = X["AveRooms"] / (X["AveOccup"] + eps)
        X["pop_per_occup"] = X["Population"] / (X["AveOccup"] + eps)
        X["latlong_interact"] = X["Latitude"] * X["Longitude"]
        if self.passthrough:
            return X.values
        else:
            return X[["bedrooms_per_room", "rooms_per_person", "pop_per_occup", "latlong_interact"]].values

    @staticmethod
    def _as_df(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=[
            'MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude'
        ])

def build_preprocessor(config: PreprocessConfig) -> Pipeline:
    """Create a preprocessing pipeline for numeric features."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if config.clip_lower_q is not None and config.clip_upper_q is not None:
        steps.append(("clipper", QuantileClipper(config.clip_lower_q, config.clip_upper_q)))
    steps.append(("feateng", FeatureEngineer()))

    # Optional nonlinear transform
    if config.nonlin == "power":
        steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))
    elif config.nonlin == "quantile":
        steps.append(("qtf", QuantileTransformer(output_distribution="normal")))

    # Scaling
    if config.scaler == "standard":
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    elif config.scaler == "robust":
        steps.append(("scaler", RobustScaler(with_centering=True, with_scaling=True)))

    return Pipeline(steps)

def load_data(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Load California Housing data as (X, y)."""
    ds = fetch_california_housing(as_frame=as_frame)
    X = ds.data.copy()
    y = ds.target.copy()
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, config: PreprocessConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=config.test_size, random_state=config.random_state)

def target_transform(y: np.ndarray, log1p: bool) -> Tuple[np.ndarray, Optional[Any]]:
    """Optionally apply log1p to target; return transformed y and inverse function."""
    if log1p:
        y_t = np.log1p(y)
        inv = np.expm1
        return y_t, inv
    return y, None

def run_preprocessing(config: Optional[PreprocessConfig] = None) -> Dict[str, Any]:
    """Load data, split, fit preprocessor on train, transform split, and return artifacts."""
    if config is None:
        config = PreprocessConfig()

    X, y = load_data(as_frame=True)

    # Diagnostics
    diag = _diagnostics(X, "X")
    tgt_diag = pd.Series({"mean": y.mean(), "std": y.std(), "min": y.min(), "max": y.max(), "skew": pd.Series(y).skew()}, name="y_stats")

    X_train, X_valid, y_train, y_valid = split_data(X, y, config)

    prep = build_preprocessor(config)
    X_train_p = prep.fit_transform(X_train, y_train)
    X_valid_p = prep.transform(X_valid)

    y_train_t, inv = target_transform(y_train.values, config.log1p_target)
    y_valid_t = np.log1p(y_valid.values) if config.log1p_target else y_valid.values

    artifacts = {
        "config": config,
        "preprocessor": prep,
        "X_train_processed": X_train_p,
        "X_valid_processed": X_valid_p,
        "y_train_transformed": y_train_t,
        "y_valid_transformed": y_valid_t,
        "y_inverse_fn": inv,
        "feature_diagnostics": diag,
        "target_diagnostics": tgt_diag.to_frame().T,
        "train_index": getattr(X_train, "index", None),
        "valid_index": getattr(X_valid, "index", None),
    }
    return artifacts

def save_artifacts(artifacts: Dict[str, Any], out_dir: str = "artifacts") -> Path:
    """Persist preprocessor and arrays for later modeling."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts["preprocessor"], out_path / "preprocessor.joblib")
    np.save(out_path / "X_train_processed.npy", artifacts["X_train_processed"])
    np.save(out_path / "X_valid_processed.npy", artifacts["X_valid_processed"])
    np.save(out_path / "y_train.npy", artifacts["y_train_transformed"])
    np.save(out_path / "y_valid.npy", artifacts["y_valid_transformed"])

    # Diagnostics as CSV
    artifacts["feature_diagnostics"].to_csv(out_path / "feature_diagnostics.csv")
    artifacts["target_diagnostics"].to_csv(out_path / "target_diagnostics.csv", index=False)

    # Save a small README
    with open(out_path / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "Artifacts generated by california_preprocessing_complete.py\n"
            " - preprocessor.joblib: fitted sklearn Pipeline\n"
            " - X_train_processed.npy / X_valid_processed.npy: feature arrays for modeling\n"
            " - y_train.npy / y_valid.npy: (optionally log1p) targets\n"
            " - feature_diagnostics.csv / target_diagnostics.csv: dataset checks\n"
        )
    return out_path

def load_preprocessor(path: str) -> Pipeline:
    return joblib.load(path)

# -----------------------------
# CLI
# -----------------------------

def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Run comprehensive preprocessing for California Housing.")
    parser.add_argument("--no-log1p", action="store_true", help="Disable log1p target transform.")
    parser.add_argument("--scaler", choices=["standard", "robust", "none"], default="standard", help="Which scaler to use.")
    parser.add_argument("--nonlin", choices=["power", "quantile", "none"], default="none", help="Optional nonlinear transform.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--clip-lower", type=float, default=0.005, help="Lower quantile for outlier clipping.")
    parser.add_argument("--clip-upper", type=float, default=0.995, help="Upper quantile for outlier clipping.")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Where to save outputs.")
    args = parser.parse_args()

    cfg = PreprocessConfig(
        test_size=args.test_size,
        clip_lower_q=None if args.clip_lower < 0 else args.clip_lower,
        clip_upper_q=None if args.clip_upper < 0 else args.clip_upper,
        scaler=None if args.scaler == "none" else args.scaler,
        nonlin=None if args.nonlin == "none" else args.nonlin,
        log1p_target=not args.no_log1p,
    )

    arts = run_preprocessing(cfg)
    out_dir = save_artifacts(arts, args.out_dir)
    print(f"Saved to {out_dir.resolve()}")

if __name__ == "__main__":
    _cli()
