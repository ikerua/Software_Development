"""
This module contains functions and classes for making predictions with trained models.
It includes functionalities to load a trained model, run inference on test data,
and save the prediction results.
"""

__docformat__ = "numpy"

import os
import argparse
import numpy as np
import pandas as pd
import torch
from joblib import load
from typing import Tuple
from pathlib import Path


from my_project.model import HousePriceRegressor

def _resolve_paths(data_dir: str, models_dir: str, output_path: str) -> Tuple[str, str, str]:
    """
    Resolve relative paths to absolute paths based on the project root 
    and ensure the output directory exists.

    Parameters
    ----------
    data_dir : str
        Relative or absolute path to the data directory.
    models_dir : str
        Relative or absolute path to the models directory (for checkpoint).
    output_path : str
        Relative or absolute path for the prediction output file.

    Returns
    -------
    Tuple[str, str, str]
        Resolved (data_dir, models_dir, output_path) as absolute file paths.

    Examples
    --------
    >>> _resolve_paths("data/processed", "models", "models/test_predictions.csv")
    ('C:/project/data/processed', 'C:/project/models', 'C:/project/models/test_predictions.csv')
    """
    root = Path(__file__).resolve().parents[2]
    
    data_dir_abs = root / data_dir
    models_dir_abs = root / models_dir
    output_path_abs = root / output_path
    
    os.makedirs(output_path_abs.parent, exist_ok=True)
    return str(data_dir_abs), str(models_dir_abs), str(output_path_abs)


def _pick_ckpt(models_dir: str) -> str:
    """
    Locate the expected model checkpoint file.

    Parameters
    ----------
    models_dir : str
        Directory where the model checkpoints are stored.

    Returns
    -------
    str
        The absolute path to the best model checkpoint file.

    Raises
    ------
    FileNotFoundError
        If the required checkpoint file does not exist.

    Examples
    --------
    >>> _pick_ckpt("models")
    'C:/project/models/house_price_regressor.ckpt'
    """
    models_path = Path(models_dir)
    ckpt_files = list(models_path.glob("*.ckpt"))
    
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt files found in {models_dir}. Train the model first.")
    
    # Encuentra el archivo modificado más recientemente
    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Found latest checkpoint: {latest_ckpt}")
    return str(latest_ckpt)


def run_predict(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    output_path: str = "models/test_predictions.csv",
    device: str = "auto",
    target_col: str = "House_Price",
) -> str:
    """
    Run inference with a trained regression model and save predictions.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing 'test.csv' (scaled data) and 'y_scaler.joblib', 
        by default "data/processed".
    models_dir : str, optional
        Directory containing the trained model checkpoint, by default "models".
    output_path : str, optional
        Path where the resulting DataFrame (predictions and true values) 
        will be saved as a CSV, by default "models/test_predictions.csv".
    device : str, optional
        The PyTorch device to use for inference ('cpu', 'cuda', 'auto'), 
        by default "auto".
    target_col : str, optional
        The name of the target variable in the dataset, by default "House_Price".

    Returns
    -------
    str
        The path where the prediction results were saved.

    Raises
    ------
    FileNotFoundError
        If the model checkpoint or the 'test.csv' file cannot be found.

    Examples
    --------
    >>> run_predict(data_dir="data/processed", models_dir="models", output_path="models/test_predictions.csv")
    'C:/project/models/test_predictions.csv'
    """
    # Rutas
    data_dir, models_dir, output_path = _resolve_paths(data_dir, models_dir, output_path)

    # Cargar checkpoint
    ckpt_path = _pick_ckpt(models_dir)
    print(f"Using checkpoint: {ckpt_path}")
    model = HousePriceRegressor.load_from_checkpoint(ckpt_path)
    model.eval()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_csv = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"{test_csv} does not exist")
    test_df = pd.read_csv(test_csv)

    y_scaler_path = os.path.join(data_dir, "y_scaler.joblib")
    y_scaler = load(y_scaler_path) if os.path.exists(y_scaler_path) else None
    
    if y_scaler is None:
        print("Warning: y_scaler not found. Predictions will be output in scaled units.")

    # Features
    feature_names = [c for c in test_df.columns if c != target_col]
    X_test = torch.tensor(test_df[feature_names].values, dtype=torch.float32, device=device)

    # Inference
    with torch.no_grad():
        preds_scaled = model(X_test).detach().cpu().numpy().reshape(-1, 1)

    # Inverse transform
    if y_scaler is not None:
        preds = y_scaler.inverse_transform(preds_scaled).ravel()
        y_true_unscaled = None
        if target_col in test_df.columns:
            y_true_unscaled = y_scaler.inverse_transform(
                test_df[target_col].values.reshape(-1, 1)
            ).ravel()
    else:
        preds = preds_scaled.ravel()
        y_true_unscaled = test_df[target_col].values if target_col in test_df.columns else None

    out_df = pd.DataFrame({"prediction": preds})
    if y_true_unscaled is not None:
        out_df["y_true"] = y_true_unscaled
        from sklearn.metrics import mean_squared_error
        rmse = float(np.sqrt(mean_squared_error(y_true_unscaled, preds)))
        print(f"Test RMSE (original units): {rmse:.6f}")

    out_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    return output_path


def main():
    """
    Parse command-line arguments and initiate the `run_predict` function.

    Examples
    --------
    From Python:
    >>> from my_project.modeling import predict
    >>> predict.main()
    """
    parser = argparse.ArgumentParser(description="Inference using Lightning checkpoint (.ckpt)")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Folder with test.csv and scalers")
    parser.add_argument("--models_dir", type=str, default="models", help="Folder with .ckpt")
    parser.add_argument("--output_path", type=str, default="models/test_predictions.csv", help="Output CSV")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    parser.add_argument("--target_col", type=str, default="House_Price", help="Name of the target column")
    args = parser.parse_args()

    run_predict(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        output_path=args.output_path,
        device=args.device,
        target_col=args.target_col,
    )


if __name__ == "__main__":
    main()
