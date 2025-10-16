# /// script
# dependencies = [
#   "gradio",
#   "pandas",
#   "torch",
#   "pytorch-lightning",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn",
# ]
# ///

import gradio as gr
import pandas as pd
import sys
import os
import argparse
import importlib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

import my_project.dataset as ds
from my_project.modeling import train, predict

# Global variables to store state
trained_model_path = None
training_history = []

def prepare_data(data_path, train_ratio, val_ratio, test_ratio, progress=gr.Progress()):
    """Prepare the dataset for training with custom split ratios"""
    try:
        # Validate ratios sum to 1.0
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            return f"### ❌ Error: Ratios must sum to 1.0 (currently {total:.2f})", gr.update(interactive=False)
        
        progress(0, desc="Initializing data module...")
        
        dm = ds.HousePricingDataModule(
            data_dir=data_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        progress(0.5, desc="Preparing and splitting data...")
        dm.prepare_data()
        
        progress(0.8, desc="Setting up datasets...")
        # Get dataset info
        dm.setup('fit')
        train_size = len(dm.train_dataloader().dataset)
        val_size = len(dm.val_dataloader().dataset)
        
        dm.setup('test')
        test_size = len(dm.test_dataloader().dataset)
        
        total_size = train_size + val_size + test_size
        
        progress(1.0, desc="Complete!")
        
        stats = f"""
        ### ✅ Data Preparation Complete!
        
        **Dataset Configuration:**
        - **Dataset Path**: `{data_path}`
        - **Total Samples**: {total_size}
        
        **Split Configuration:**
        - **Train Ratio**: {train_ratio:.1%} → {train_size} samples
        - **Validation Ratio**: {val_ratio:.1%} → {val_size} samples
        - **Test Ratio**: {test_ratio:.1%} → {test_size} samples
        
        **Verification:**
        - Actual Train: {train_size/total_size:.1%}
        - Actual Val: {val_size/total_size:.1%}
        - Actual Test: {test_size/total_size:.1%}
        
        **Status**: ✅ Ready for training
        
        📁 Processed data saved to: `data/processed/`
        """
        
        # Create split visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        sizes = [train_size, val_size, test_size]
        labels = ['Train', 'Validation', 'Test']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.05, 0.05, 0.05)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.set_title('Data Split Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        ax2.bar(labels, sizes, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Sample Counts by Split', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(sizes):
            ax2.text(i, v + max(sizes)*0.02, str(v), ha='center', fontweight='bold')
        

        plt.tight_layout()
        
        return stats, True, fig
        
    except Exception as e:
        return f"### ❌ Error: {str(e)}", gr.update(interactive=False), None

def update_test_ratio(train_val, val_val):
    """Auto-calculate test ratio to ensure sum = 1.0"""
    test_val = 1.0 - train_val - val_val
    return gr.update(value=max(0.0, min(1.0, test_val)))

def train_model(batch_size, learning_rate, weight_decay, epochs, num_workers, progress=gr.Progress()):
    """Train the regression model"""
    global trained_model_path, training_history
    
    try:
        progress(0, desc="Starting training...")
        
        # Create arguments
        args = argparse.Namespace(
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            epochs=int(epochs),
        )
        
        # Train the model
        progress(0.3, desc="Training model...")
        train.main(args)
        
        # Find the latest model
        models_dir = Path("../../models")
        model_files = list(models_dir.glob("*.ckpt"))
        if model_files:
            trained_model_path = str(max(model_files, key=os.path.getctime))
        
        progress(1.0, desc="Training complete!")
        
        stats = f"""
        ### ✅ Training Complete!
        
        **Training Configuration:**
        - **Batch Size**: {batch_size}
        - **Learning Rate**: {learning_rate}
        - **Weight Decay**: {weight_decay}
        - **Epochs**: {epochs}
        - **Workers**: {num_workers}
        
        **Model saved to**: `{trained_model_path}`
        
        🎉 Model is ready for predictions!
        """
        
        return stats, True, gr.update(visible=True)
        
    except Exception as e:
        return f"### ❌ Training Error: {str(e)}", gr.update(interactive=False), gr.update(visible=False)

def make_predictions(data_dir, models_dir, output_path, target_col, progress=gr.Progress()):
    """Make predictions using the trained model"""
    try:
        progress(0, desc="Loading model...")
        
        # Run predictions
        progress(0.5, desc="Making predictions...")
        output_csv = predict.run_predict(
            data_dir=data_dir,
            models_dir=models_dir,
            output_path=output_path,
            device="auto",
            target_col=target_col,
        )
        
        progress(0.8, desc="Loading results...")
        
        # Load and display predictions
        df_preds = pd.read_csv(output_csv)
        
        progress(1.0, desc="Complete!")
        
        stats = f"""
        ### ✅ Predictions Complete!
        
        - **Output File**: `{output_csv}`
        - **Total Predictions**: {len(df_preds)}
        - **Columns**: {', '.join(df_preds.columns.tolist())}
        
        📊 Preview shown below
        """
        
        return stats, df_preds.head(20), output_csv
        
    except Exception as e:
        return f"### ❌ Prediction Error: {str(e)}", None, None

def analyze_predictions(csv_path):
    """Analyze prediction results"""
    try:
        df = pd.read_csv(csv_path)
        
        # Check if predictions and actual values exist
        pred_col = 'predicted_House_Price' if 'predicted_House_Price' in df.columns else 'prediction'
        actual_col = 'House_Price' if 'House_Price' in df.columns else 'actual'
        
        if pred_col not in df.columns or actual_col not in df.columns:
            return "Columns not found. Available columns: " + ", ".join(df.columns), None, None
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        mae = mean_absolute_error(df[actual_col], df[pred_col])
        mse = mean_squared_error(df[actual_col], df[pred_col])
        rmse = np.sqrt(mse)
        r2 = r2_score(df[actual_col], df[pred_col])
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot: Actual vs Predicted
        axes[0].scatter(df[actual_col], df[pred_col], alpha=0.5)
        axes[0].plot([df[actual_col].min(), df[actual_col].max()], 
                     [df[actual_col].min(), df[actual_col].max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual House Price', fontsize=12)
        axes[0].set_ylabel('Predicted House Price', fontsize=12)
        axes[0].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = df[actual_col] - df[pred_col]
        axes[1].scatter(df[pred_col], residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted House Price', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Distribution comparison
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[actual_col], bins=30, alpha=0.5, label='Actual', color='blue')
        ax.hist(df[pred_col], bins=30, alpha=0.5, label='Predicted', color='orange')
        ax.set_xlabel('House Price', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution: Actual vs Predicted', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        metrics_text = f"""
        ### 📊 Model Performance Metrics
        
        - **Mean Absolute Error (MAE)**: ${mae:,.2f}
        - **Root Mean Squared Error (RMSE)**: ${rmse:,.2f}
        - **R² Score**: {r2:.4f}
        - **Mean Squared Error (MSE)**: ${mse:,.2f}
        
        {'🎉 Excellent!' if r2 > 0.9 else '✅ Good!' if r2 > 0.7 else '⚠️ Needs improvement'}
        """
        
        return metrics_text, fig, fig2
        
    except Exception as e:
        return f"### ❌ Analysis Error: {str(e)}", None, None

def load_and_preview_data(file_path):
    """Load and preview dataset"""
    try:
        df = pd.read_csv(file_path)
        
        stats = f"""
        ### 📋 Dataset Preview
        
        - **Total Rows**: {len(df)}
        - **Total Columns**: {len(df.columns)}
        - **Columns**: {', '.join(df.columns.tolist())}
        - **Missing Values**: {df.isnull().sum().sum()}
        """
        
        # Basic statistics
        desc = df.describe()
        
        return stats, df.head(10), desc
        
    except Exception as e:
        return f"### ❌ Error: {str(e)}", None, None

# Create Gradio Interface
with gr.Blocks(title="House Price Model Training & Prediction", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏠 House Price Regression - Model Training & Prediction Pipeline
        
        Complete workflow for data preparation, model training, prediction, and analysis.
        Follow the tabs in order for the full pipeline.
        """
    )
    
    with gr.Tabs():
        # Tab 1: Data Preparation
        with gr.Tab("📁 1. Data Preparation"):
            gr.Markdown(
                """
                ### Step 1: Load and Prepare Your Dataset
                
                Provide the path to your raw dataset CSV file and configure the train/val/test split.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    data_path_input = gr.Textbox(
                        value="../../data/raw/house_price_regression_dataset.csv",
                        label="Dataset Path",
                        placeholder="Path to your CSV file"
                    )
                    
                    gr.Markdown("#### 📊 Configure Data Split Ratios")
                    gr.Markdown("*Adjust the sliders to set train/validation/test proportions (must sum to 1.0)*")
                    
                    train_ratio_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.6,
                        step=0.05,
                        label="Training Set Ratio",
                        info="Proportion of data for training"
                    )
                    
                    val_ratio_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.5,
                        value=0.2,
                        step=0.05,
                        label="Validation Set Ratio",
                        info="Proportion of data for validation"
                    )
                    
                    test_ratio_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.5,
                        value=0.2,
                        step=0.05,
                        label="Test Set Ratio",
                        info="Proportion of data for testing (auto-calculated)",
                        interactive=True
                    )
                    
                    prepare_btn = gr.Button("📊 Prepare Data", variant="primary", size="lg")
                    
                    gr.Markdown("#### 👁️ Preview Dataset (Optional)")
                    preview_btn = gr.Button("Preview Raw Data")
                
                with gr.Column(scale=2):
                    prep_output = gr.Markdown()
                    split_plot = gr.Plot(label="Data Split Visualization")
                    preview_stats = gr.Markdown()
                    preview_df = gr.Dataframe(label="Data Preview")
                    preview_desc = gr.Dataframe(label="Statistical Summary")
            
            train_btn_enable = gr.State(value=False)
            
            # Auto-update test ratio when train or val changes
            train_ratio_slider.change(
                update_test_ratio,
                inputs=[train_ratio_slider, val_ratio_slider],
                outputs=[test_ratio_slider]
            )
            
            val_ratio_slider.change(
                update_test_ratio,
                inputs=[train_ratio_slider, val_ratio_slider],
                outputs=[test_ratio_slider]
            )
            
            prepare_btn.click(
                prepare_data,
                inputs=[data_path_input, train_ratio_slider, val_ratio_slider, test_ratio_slider],
                outputs=[prep_output, train_btn_enable, split_plot]
            )
            
            preview_btn.click(
                load_and_preview_data,
                inputs=[data_path_input],
                outputs=[preview_stats, preview_df, preview_desc]
            )
        
        # Tab 2: Model Training
        with gr.Tab("🎯 2. Model Training"):
            gr.Markdown(
                """
                ### Step 2: Train the Regression Model
                
                Configure training parameters and start training. Monitor the process and results.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Training Hyperparameters")
                    
                    batch_size_slider = gr.Slider(
                        minimum=8,
                        maximum=256,
                        value=64,
                        step=8,
                        label="Batch Size"
                    )
                    
                    lr_slider = gr.Slider(
                        minimum=1e-5,
                        maximum=1e-1,
                        value=1e-3,
                        step=1e-5,
                        label="Learning Rate"
                    )
                    
                    weight_decay_slider = gr.Slider(
                        minimum=0.0,
                        maximum=0.1,
                        value=0.0,
                        step=0.001,
                        label="Weight Decay (L2 Regularization)"
                    )
                    
                    epochs_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1,
                        label="Number of Epochs"
                    )
                    
                    workers_slider = gr.Slider(
                        minimum=0,
                        maximum=16,
                        value=4,
                        step=1,
                        label="Number of Workers"
                    )
                    
                    train_start_btn = gr.Button(
                        "🚀 Start Training",
                        variant="primary",
                        size="lg",
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    train_output = gr.Markdown()
                    training_plot = gr.Plot(label="Training Progress", visible=False)
            
            predict_btn_enable = gr.State(value=False)
            
            # Link train button to preparation state
            train_btn_enable.change(
                lambda x: gr.update(interactive=x),
                inputs=[train_btn_enable],
                outputs=[train_start_btn]
            )
            
            train_start_btn.click(
                train_model,
                inputs=[batch_size_slider, lr_slider, weight_decay_slider, epochs_slider, workers_slider],
                outputs=[train_output, predict_btn_enable, training_plot]
            )
        
        # Tab 3: Predictions
        with gr.Tab("🔮 3. Make Predictions"):
            gr.Markdown(
                """
                ### Step 3: Generate Predictions
                
                Use the trained model to make predictions on the test dataset.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Prediction Configuration")
                    
                    pred_data_dir = gr.Textbox(
                        value="data/processed",
                        label="Processed Data Directory"
                    )
                    
                    pred_models_dir = gr.Textbox(
                        value="models",
                        label="Models Directory"
                    )
                    
                    pred_output_path = gr.Textbox(
                        value="models/test_predictions.csv",
                        label="Output CSV Path"
                    )
                    
                    pred_target_col = gr.Textbox(
                        value="House_Price",
                        label="Target Column Name"
                    )
                    
                    predict_start_btn = gr.Button(
                        "🔮 Generate Predictions",
                        variant="primary",
                        size="lg",
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    pred_output = gr.Markdown()
                    pred_df = gr.Dataframe(label="Predictions Preview")
            
            pred_file_path = gr.State()
            
            # Link predict button to training state
            predict_btn_enable.change(
                lambda x: gr.update(interactive=x),
                inputs=[predict_btn_enable],
                outputs=[predict_start_btn]
            )
            
            predict_start_btn.click(
                make_predictions,
                inputs=[pred_data_dir, pred_models_dir, pred_output_path, pred_target_col],
                outputs=[pred_output, pred_df, pred_file_path]
            )        
    
    gr.Markdown(
        """
        ---
        ### 📚 Pipeline Summary
        
        1. **Data Preparation**: Load and preprocess your dataset with custom train/val/test splits
        2. **Model Training**: Configure hyperparameters and train the regression model
        3. **Make Predictions**: Generate predictions on test data
        
        **Tip**: Follow the tabs in order for a complete machine learning workflow!
        
        ---
        *Current User*: joaquinorradreahora | *Date*: 2025-10-16 09:12:32
        """
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7861)