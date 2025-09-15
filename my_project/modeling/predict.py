# my_project/modeling/predict.py
import os
import argparse
import glob
import numpy as np
import pandas as pd
import torch

from my_project.model import HousePriceRegressor

def _resolve_paths(data_dir: str, models_dir: str, output_path: str):
    # Get absolute paths for data, models, and output
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.abspath(os.path.join(root, data_dir)) if not os.path.isabs(data_dir) else data_dir
    models_dir = os.path.abspath(os.path.join(root, models_dir)) if not os.path.isabs(models_dir) else models_dir
    output_path = os.path.abspath(os.path.join(root, output_path)) if not os.path.isabs(output_path) else output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return data_dir, models_dir, output_path

def _pick_ckpt(models_dir: str) -> str:
    exact = os.path.join(models_dir, "house_price_regressor.ckpt")
    if os.path.exists(exact):
        return exact
    else:
        raise FileNotFoundError(f"{exact} does not exist.")

def run_predict(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    output_path: str = "models/test_predictions.csv",
    device: str = "auto",
    target_col: str = "House_Price",
):
    # Get absolute paths
    data_dir, models_dir, output_path = _resolve_paths(data_dir, models_dir, output_path)

    # Load checkpoint
    ckpt_path = _pick_ckpt(models_dir)
    print(f"Using checkpoint: {ckpt_path}")

    model = HousePriceRegressor.load_from_checkpoint(ckpt_path)
    model.eval()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load test data
    test_csv = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"{test_csv} does not exist")
    test_df = pd.read_csv(test_csv)

    # Select features
    feature_names = [c for c in test_df.columns if c != target_col]
    X_test = torch.tensor(test_df[feature_names].values, dtype=torch.float32, device=device)

    # Inference
    with torch.no_grad():
        preds = model(X_test).detach().cpu().numpy()

    out_df = pd.DataFrame({"prediction": preds})

    if target_col in test_df.columns:
        from sklearn.metrics import mean_squared_error
        rmse = float(np.sqrt(mean_squared_error(test_df[target_col].values, preds)))
        out_df["y_true"] = test_df[target_col].values
        print(f"Test RMSE: {rmse:.6f}")

    out_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Inference using Lightning checkpoint (.ckpt)")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Folder with test.csv")
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
