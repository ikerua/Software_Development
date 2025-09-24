# my_project/modeling/predict.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
from joblib import load

from my_project.model import HousePriceRegressor


def _resolve_paths(data_dir: str, models_dir: str, output_path: str):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.abspath(os.path.join(root, data_dir)) if not os.path.isabs(data_dir) else data_dir
    models_dir = os.path.abspath(os.path.join(root, models_dir)) if not os.path.isabs(models_dir) else models_dir
    output_path = os.path.abspath(os.path.join(root, output_path)) if not os.path.isabs(output_path) else output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return data_dir, models_dir, output_path


def _pick_ckpt(models_dir: str) -> str:
    ckpt_path = os.path.join(models_dir, "house_price_regressor.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} does not exist. Train the model first.")
    return ckpt_path


def run_predict(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    output_path: str = "models/test_predictions.csv",
    device: str = "auto",
    target_col: str = "House_Price",
):
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

    # Features
    feature_names = [c for c in test_df.columns if c != target_col]
    X_test = torch.tensor(test_df[feature_names].values, dtype=torch.float32, device=device)

    with torch.no_grad():
        preds_scaled = model(X_test).detach().cpu().numpy().reshape(-1, 1)

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
    parser = argparse.ArgumentParser(description="Inference using Lightning checkpoint (.ckpt)")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Folder with test.csv and scalers")
    parser.add_argument("--models_dir", type=str, default="models", help="Folder with .ckpt")
    parser.add_argument("--output_path", type=str, default="models/test_predictions.csv", help="Output CSV")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    parser.add_argument("--target_col", type=str, default="House_Price")
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
