# /// script
# dependencies = [
#   "gradio",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn",
# ]
# ///

import gradio as gr
import os
import sys
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup styling
sns.set_theme(context="notebook", style="whitegrid")

ROOT = Path(os.getcwd()).parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Metric calculation functions
def rmse(y, yhat): 
    return float(np.sqrt(np.mean((y - yhat)**2)))

def mae(y, yhat):  
    return float(np.mean(np.abs(y - yhat)))

def r2(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan

def mape(y, yhat, eps=1e-8):
    denom = np.maximum(np.abs(y), eps)
    return float(np.mean(np.abs((y - yhat) / denom))) * 100.0

def load_predictions(pred_path):
    """Load and validate predictions file"""
    try:
        pred_path = Path(pred_path)
        if not pred_path.exists():
            return None, f"❌ File not found: {pred_path}"
        
        pred_df = pd.read_csv(pred_path)
        
        # Validate required columns
        required = ["y_true", "prediction"]
        missing = [col for col in required if col not in pred_df.columns]
        if missing:
            return None, f"❌ Missing columns: {missing}. Available: {pred_df.columns.tolist()}"
        
        return pred_df, None
    except Exception as e:
        return None, f"❌ Error loading file: {str(e)}"

def calculate_metrics(pred_path):
    """Calculate all metrics and display summary"""
    pred_df, error = load_predictions(pred_path)
    if error:
        return error, None
    
    y = pred_df["y_true"].to_numpy()
    yhat = pred_df["prediction"].to_numpy()
    
    metrics = {
        "RMSE": rmse(y, yhat),
        "MAE": mae(y, yhat),
        "R2": r2(y, yhat),
        "MAPE(%)": mape(y, yhat)
    }
    
    # Performance rating
    r2_val = metrics["R2"]
    if r2_val > 0.95:
        rating = "🎉 Excellent!"
        color = "green"
    elif r2_val > 0.85:
        rating = "✅ Very Good!"
        color = "lightgreen"
    elif r2_val > 0.7:
        rating = "👍 Good"
        color = "yellow"
    else:
        rating = "⚠️ Needs Improvement"
        color = "orange"
    
    metrics_text = f"""
    ### 📊 Model Performance Metrics
    
    **Overall Rating**: {rating}
    
    **Regression Metrics:**
    - **R² Score**: {metrics['R2']:.4f}
    - **RMSE**: {metrics['RMSE']:.2f}
    - **MAE**: {metrics['MAE']:.2f}
    - **MAPE**: {metrics['MAPE(%)']:.2f}%
    
    **Dataset Info:**
    - **Total Samples**: {len(pred_df)}
    - **Predictions File**: `{pred_path}`
    
    ---
    **JSON Export:**
    ```json
    {json.dumps(metrics, indent=2)}
    ```
    """
    
    return metrics_text, pred_df.head(20)

def plot_predictions_vs_actual(pred_path, save_fig):
    """Scatter plot: Actual vs Predicted"""
    pred_df, error = load_predictions(pred_path)
    if error:
        return None, error
    
    try:
        y = pred_df["y_true"].to_numpy()
        yhat = pred_df["prediction"].to_numpy()
        
        metrics = {
            "RMSE": rmse(y, yhat),
            "MAE": mae(y, yhat),
            "R2": r2(y, yhat)
        }
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(pred_df["y_true"], pred_df["prediction"], alpha=0.6, s=40, edgecolors='k', linewidths=0.5)
        
        # Perfect prediction line
        lo = float(min(pred_df["y_true"].min(), pred_df["prediction"].min()))
        hi = float(max(pred_df["y_true"].max(), pred_df["prediction"].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2, color="red", label="Perfect Prediction")
        
        ax.set_xlabel("Actual (y)", fontsize=12)
        ax.set_ylabel("Prediction (ŷ)", fontsize=12)
        ax.set_title(
            f"Predictions vs Actuals\n"
            f"RMSE={metrics['RMSE']:.2f}  •  MAE={metrics['MAE']:.2f}  •  R²={metrics['R2']:.4f}",
            fontsize=14, fontweight='bold'
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            save_path = FIG_DIR / "pred_vs_actual.svg"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            status = f"✅ Saved to: {save_path}"
        else:
            status = "📊 Plot generated (not saved)"
        
        return fig, status
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def plot_training_validation_loss(log_dir, save_fig):
    """Plot training and validation loss curves"""
    try:
        metrics_path = Path(log_dir) / "metrics.csv"
        
        if not metrics_path.exists():
            return None, f"❌ Metrics file not found: {metrics_path}"
        
        df = pd.read_csv(metrics_path)
        
        # Extract losses
        train_loss = df[df["train_loss"].notna()][["epoch", "train_loss"]]
        val_loss = df[df["val_loss"].notna()][["epoch", "val_loss"]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_loss["epoch"], train_loss["train_loss"], 
                marker='o', label="Train Loss", linewidth=2, markersize=5)
        ax.plot(val_loss["epoch"], val_loss["val_loss"], 
                marker='s', label="Validation Loss", linewidth=2, markersize=5)
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (MSE)", fontsize=12)
        ax.set_title("Training vs Validation Loss", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            save_path = FIG_DIR / "training_vs_validation_loss.svg"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            status = f"✅ Saved to: {save_path}"
        else:
            status = "📊 Plot generated (not saved)"
        
        return fig, status
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def plot_residuals(pred_path, bins, save_fig):
    """Plot residuals distribution"""
    pred_df, error = load_predictions(pred_path)
    if error:
        return None, error
    
    try:
        resid = pred_df["y_true"] - pred_df["prediction"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE
        ax1.hist(resid, bins=bins, alpha=0.7, edgecolor='black', color='skyblue')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.set_xlabel("Residual (y - ŷ)", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("Residuals Distribution", fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Q-Q Plot for normality check
        from scipy import stats
        stats.probplot(resid, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normality Check)", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Calculate statistics
        mean_resid = resid.mean()
        std_resid = resid.std()
        
        stats_text = f"""
        **Residual Statistics:**
        - Mean: {mean_resid:.4f}
        - Std Dev: {std_resid:.4f}
        - Min: {resid.min():.4f}
        - Max: {resid.max():.4f}
        """
        
        if save_fig:
            save_path = FIG_DIR / "residuals_hist.svg"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            status = f"✅ Saved to: {save_path}\n\n{stats_text}"
        else:
            status = f"📊 Plot generated (not saved)\n\n{stats_text}"
        
        return fig, status
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def plot_worst_predictions(pred_path, top_n, save_fig):
    """Plot worst predictions by MSE"""
    pred_df, error = load_predictions(pred_path)
    if error:
        return None, None, error
    
    try:
        pred_df["abs_error"] = (pred_df["y_true"] - pred_df["prediction"]).abs()
        pred_df["mse_sample"] = (pred_df["y_true"] - pred_df["prediction"])**2
        
        worst_mse = pred_df.sort_values("mse_sample", ascending=False).head(top_n).reset_index(drop=True)
        
        # Create bar plot
        x = np.arange(len(worst_mse))
        width = 0.4
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.bar(x - width/2, worst_mse["abs_error"], width=width, label="Absolute Error", alpha=0.8)
        ax.bar(x + width/2, worst_mse["mse_sample"], width=width, label="MSE Loss", alpha=0.8)
        
        ax.set_xlabel(f"Sample (Top-{top_n} by MSE)", fontsize=12)
        ax.set_ylabel("Error Value", fontsize=12)
        ax.set_title(f"Top-{top_n} Worst Predictions: Absolute Error vs MSE Loss", 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.legend(title="Metric", fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_fig:
            save_path = FIG_DIR / "absolute_error_vs_mse.svg"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            status = f"✅ Saved to: {save_path}"
        else:
            status = "📊 Plot generated (not saved)"
        
        # Return dataframe for display
        display_df = worst_mse[["y_true", "prediction", "abs_error", "mse_sample"]].round(4)
        
        return fig, display_df, status
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

def plot_error_by_quantiles(pred_path, n_quantiles, save_fig):
    """Plot error by target quantiles"""
    pred_df, error = load_predictions(pred_path)
    if error:
        return None, None, error
    
    try:
        bins = pd.qcut(pred_df["y_true"], q=n_quantiles, duplicates="drop")
        
        grouped = pred_df.assign(bin=bins).groupby("bin").apply(
            lambda g: pd.Series({
                "count": len(g),
                "RMSE": rmse(g["y_true"].to_numpy(), g["prediction"].to_numpy()),
                "MAE": mae(g["y_true"].to_numpy(), g["prediction"].to_numpy()),
            })
        ).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(grouped))
        ax.plot(x, grouped["RMSE"], marker="o", linewidth=2, markersize=8, label="RMSE")
        ax.plot(x, grouped["MAE"], marker="s", linewidth=2, markersize=8, label="MAE")
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i+1}" for i in x], rotation=0)
        ax.set_xlabel(f"Target Quantile Bins (n={n_quantiles})", fontsize=12)
        ax.set_ylabel("Error", fontsize=12)
        ax.set_title("Error by Target Quantiles", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            save_path = FIG_DIR / "error_by_target_quantiles.svg"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            status = f"✅ Saved to: {save_path}"
        else:
            status = "📊 Plot generated (not saved)"
        
        return fig, grouped, status
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

def plot_error_vs_features(pred_path, test_path, top_n, save_fig):
    """Plot error vs top correlated features"""
    pred_df, error = load_predictions(pred_path)
    if error:
        return None, None, error
    
    try:
        test_path = Path(test_path)
        if not test_path.exists():
            return None, None, f"❌ Test file not found: {test_path}"
        
        test_df = pd.read_csv(test_path)
        
        if len(test_df) != len(pred_df):
            return None, None, f"❌ Length mismatch: test={len(test_df)}, pred={len(pred_df)}"
        
        merged = test_df.copy()
        merged["y_true"] = pred_df["y_true"].values
        merged["prediction"] = pred_df["prediction"].values
        merged["abs_err"] = (merged["y_true"] - merged["prediction"]).abs()
        
        # Get numeric columns
        num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        for col in ["House_Price", "y_true", "prediction", "abs_err"]:
            if col in num_cols:
                num_cols.remove(col)
        
        if not num_cols:
            return None, None, "❌ No numeric features found"
        
        # Calculate correlations
        corr = merged[num_cols + ["abs_err"]].corr(numeric_only=True)["abs_err"].drop("abs_err").sort_values(ascending=False)
        
        top_show = corr.head(min(top_n, len(corr))).index.tolist()
        
        if not top_show:
            return None, None, "❌ No features to plot"
        
        # Create subplots
        n = len(top_show)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
        axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else [axes]
        
        for i, col in enumerate(top_show):
            axes[i].scatter(merged[col], merged["abs_err"], alpha=0.4, s=30)
            axes[i].set_xlabel(col, fontsize=11)
            axes[i].set_ylabel("|error|", fontsize=11)
            axes[i].set_title(f"|error| vs {col}\n(corr={corr[col]:.3f})", fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        
        plt.tight_layout()
        
        if save_fig:
            save_path = FIG_DIR / "abs_error_vs_top_features.svg"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            status = f"✅ Saved to: {save_path}"
        else:
            status = "📊 Plot generated (not saved)"
        
        # Return correlation data
        corr_df = pd.DataFrame({
            'Feature': corr.index,
            'Correlation with |error|': corr.values
        }).head(10)
        
        return fig, corr_df, status
        
    except Exception as e:
        import traceback
        return None, None, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"

# Create Gradio Interface
with gr.Blocks(title="Model Performance Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📊 House Price Model - Performance Analysis
        
        Comprehensive analysis of model predictions with interactive visualizations.
        All plots can be saved to `reports/figures/` for documentation.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 File Paths Configuration")
            pred_path_input = gr.Textbox(
                value="../models/test_predictions.csv",
                label="Predictions CSV Path",
                placeholder="Path to test_predictions.csv"
            )
            test_path_input = gr.Textbox(
                value="../data/processed/test.csv",
                label="Test Data CSV Path (for feature analysis)",
                placeholder="Path to test.csv"
            )
            log_dir_input = gr.Textbox(
                value="../reports/logs/house_price/version_1",
                label="Training Logs Directory",
                placeholder="Path to metrics.csv folder"
            )
    
    with gr.Tabs():
        # Tab 1: Metrics Overview
        with gr.Tab("📊 1. Metrics Overview"):
            gr.Markdown("### Calculate and display all performance metrics")
            
            calc_metrics_btn = gr.Button("📈 Calculate Metrics", variant="primary", size="lg")
            
            metrics_output = gr.Markdown()
            metrics_df = gr.Dataframe(label="Predictions Preview (First 20 rows)")
            
            calc_metrics_btn.click(
                calculate_metrics,
                inputs=[pred_path_input],
                outputs=[metrics_output, metrics_df]
            )
        
        # Tab 2: Predictions vs Actual
        with gr.Tab("🎯 2. Predictions vs Actual"):
            gr.Markdown("### Scatter plot comparing actual and predicted values")
            
            with gr.Row():
                save_pred_actual = gr.Checkbox(value=True, label="Save figure to reports/figures/")
                plot_pred_actual_btn = gr.Button("📈 Generate Plot", variant="primary")
            
            pred_actual_status = gr.Markdown()
            pred_actual_plot = gr.Plot(label="Predictions vs Actual")
            
            plot_pred_actual_btn.click(
                plot_predictions_vs_actual,
                inputs=[pred_path_input, save_pred_actual],
                outputs=[pred_actual_plot, pred_actual_status]
            )
        
        # Tab 3: Training Loss
        with gr.Tab("📉 3. Training & Validation Loss"):
            gr.Markdown("### Training and validation loss curves over epochs")
            
            with gr.Row():
                save_loss = gr.Checkbox(value=True, label="Save figure to reports/figures/")
                plot_loss_btn = gr.Button("📈 Generate Plot", variant="primary")
            
            loss_status = gr.Markdown()
            loss_plot = gr.Plot(label="Loss Curves")
            
            plot_loss_btn.click(
                plot_training_validation_loss,
                inputs=[log_dir_input, save_loss],
                outputs=[loss_plot, loss_status]
            )
        
        # Tab 4: Residuals Analysis
        with gr.Tab("📐 4. Residuals Analysis"):
            gr.Markdown("### Distribution and normality of prediction residuals")
            
            with gr.Row():
                residual_bins = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=40,
                    step=5,
                    label="Number of Bins"
                )
                save_residuals = gr.Checkbox(value=True, label="Save figure")
            
            plot_residuals_btn = gr.Button("📈 Generate Plot", variant="primary")
            
            residuals_status = gr.Markdown()
            residuals_plot = gr.Plot(label="Residuals Distribution & Q-Q Plot")
            
            plot_residuals_btn.click(
                plot_residuals,
                inputs=[pred_path_input, residual_bins, save_residuals],
                outputs=[residuals_plot, residuals_status]
            )
        
        # Tab 5: Worst Predictions
        with gr.Tab("⚠️ 5. Worst Predictions"):
            gr.Markdown("### Analyze samples with highest prediction errors")
            
            with gr.Row():
                worst_n = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Number of Worst Samples to Show"
                )
                save_worst = gr.Checkbox(value=True, label="Save figure")
            
            plot_worst_btn = gr.Button("📈 Generate Plot", variant="primary")
            
            worst_status = gr.Markdown()
            worst_plot = gr.Plot(label="Worst Predictions")
            worst_df = gr.Dataframe(label="Worst Predictions Data")
            
            plot_worst_btn.click(
                plot_worst_predictions,
                inputs=[pred_path_input, worst_n, save_worst],
                outputs=[worst_plot, worst_df, worst_status]
            )
        
        # Tab 6: Error by Quantiles
        with gr.Tab("📊 6. Error by Target Quantiles"):
            gr.Markdown("### Error distribution across different target value ranges")
            
            with gr.Row():
                n_quantiles = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Quantiles"
                )
                save_quantiles = gr.Checkbox(value=True, label="Save figure")
            
            plot_quantiles_btn = gr.Button("📈 Generate Plot", variant="primary")
            
            quantiles_status = gr.Markdown()
            quantiles_plot = gr.Plot(label="Error by Quantiles")
            quantiles_df = gr.Dataframe(label="Quantile Statistics")
            
            plot_quantiles_btn.click(
                plot_error_by_quantiles,
                inputs=[pred_path_input, n_quantiles, save_quantiles],
                outputs=[quantiles_plot, quantiles_df, quantiles_status]
            )
        
        # Tab 7: Error vs Features
        with gr.Tab("🔍 7. Error vs Top Features"):
            gr.Markdown("### Correlation between prediction error and input features")
            
            with gr.Row():
                top_features_n = gr.Slider(
                    minimum=3,
                    maximum=9,
                    value=6,
                    step=1,
                    label="Number of Top Features to Show"
                )
                save_features = gr.Checkbox(value=True, label="Save figure")
            
            plot_features_btn = gr.Button("📈 Generate Plot", variant="primary")
            
            features_status = gr.Markdown()
            features_plot = gr.Plot(label="Error vs Features")
            features_corr = gr.Dataframe(label="Feature Correlations with |error|")
            
            plot_features_btn.click(
                plot_error_vs_features,
                inputs=[pred_path_input, test_path_input, top_features_n, save_features],
                outputs=[features_plot, features_corr, features_status]
            )
    
    gr.Markdown(
        """
        ---
        ### 📚 Analysis Summary
        
        This tool provides comprehensive model performance analysis including:
        1. **Metrics Overview**: RMSE, MAE, R², MAPE
        2. **Predictions Scatter**: Visual comparison with perfect prediction line
        3. **Training Curves**: Loss evolution during training
        4. **Residuals**: Distribution and normality checks
        5. **Worst Predictions**: Identify problematic samples
        6. **Quantile Analysis**: Performance across value ranges
        7. **Feature Correlation**: Which features correlate with errors
        
        **Note**: All figures are saved as SVG for high-quality reports.
        
        ---
        *User*: joaquinorradre | *Date*: 2025-10-16 09:35:58
        """
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7862)