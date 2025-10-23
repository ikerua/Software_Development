"""
Training Pipeline Module for Gradio Interface

This module provides a complete training pipeline interface for house price regression
models through Gradio. It handles data preparation, model training configuration,
and prediction generation with interactive controls and visualizations.
"""

__docformat__ = "numpy"

import importlib.resources
import gradio as gr
import pandas as pd
import sys
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

import my_project.dataset as ds
from my_project.modeling import train, predict

# Global variable
trained_model_path = None

def prepare_data(data_path, train_ratio, val_ratio, test_ratio, progress=gr.Progress()):
    """
    Prepare the dataset for training with custom split ratios.

    Validates split ratios, initializes the data module, splits the dataset,
    and creates visualizations of the data distribution.

    Parameters
    ----------
    data_path : str
        Path to the raw CSV dataset file.
    train_ratio : float
        Proportion of data to use for training (0.0 to 1.0).
    val_ratio : float
        Proportion of data to use for validation (0.0 to 1.0).
    test_ratio : float
        Proportion of data to use for testing (0.0 to 1.0).
    progress : gr.Progress, optional
        Gradio progress tracker for UI updates, by default gr.Progress().

    Returns
    -------
    tuple of (str, bool, matplotlib.figure.Figure or None)
        Returns (status_message, success_flag, visualization_figure).
        If successful, status_message contains formatted statistics,
        success_flag is True, and figure shows data split visualization.
        On error, returns (error_message, False, None).

    Examples
    --------
    >>> stats, success, fig = prepare_data(
    ...     "data/raw/dataset.csv", 0.6, 0.2, 0.2
    ... )
    >>> if success:
    ...     print(stats)
    ...     plt.show()
    """
    try:
        # Validate ratios sum to 1.0
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            return f"### ❌ Error: Ratios must sum to 1.0 (currently {total:.2f})", False, None
        
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
        return f"### ❌ Error: {str(e)}", False, None

def update_test_ratio(train_val, val_val):
    """
    Auto-calculate test ratio to ensure sum equals 1.0.

    Parameters
    ----------
    train_val : float
        Training set ratio (0.0 to 1.0).
    val_val : float
        Validation set ratio (0.0 to 1.0).

    Returns
    -------
    gr.update
        Gradio update object with calculated test ratio value,
        clamped between 0.0 and 1.0.

    Examples
    --------
    >>> update = update_test_ratio(0.6, 0.2)
    >>> print(update)  # Should show test ratio of 0.2
    """
    test_val = 1.0 - train_val - val_val
    return gr.update(value=max(0.0, min(1.0, test_val)))

def show_training_message(batch_size, learning_rate, weight_decay, epochs, num_workers):
    """
    Display initial training configuration message.

    Parameters
    ----------
    batch_size : int
        Number of samples per training batch.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        L2 regularization weight decay parameter.
    epochs : int
        Number of training epochs.
    num_workers : int
        Number of worker processes for data loading.

    Returns
    -------
    tuple of (str, gr.update)
        Returns (formatted_message, visibility_update) showing training
        configuration and making the status message visible.

    Examples
    --------
    >>> msg, update = show_training_message(64, 0.001, 0.0, 20, 4)
    >>> print(msg)
    """
    msg = f"""
    ### 🏋️ Training in Progress...
    
    **Configuration:**
    - **Batch Size**: {batch_size}
    - **Learning Rate**: {learning_rate}
    - **Weight Decay**: {weight_decay}
    - **Epochs**: {epochs}
    - **Workers**: {num_workers}
    
    ⏳ **Please wait... Training the model...**
    
    _This may take several minutes. Do not close this window._
    """
    return msg, gr.update(visible=True)

def train_model(batch_size, learning_rate, weight_decay, epochs, num_workers):
    """
    Train the regression model with specified hyperparameters.

    Creates training arguments, executes the training pipeline, and
    updates the global trained model path.

    Parameters
    ----------
    batch_size : int
        Number of samples per training batch.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        L2 regularization weight decay parameter.
    epochs : int
        Number of training epochs.
    num_workers : int
        Number of worker processes for data loading.

    Returns
    -------
    tuple of (str, bool, gr.update)
        Returns (status_message, success_flag, visibility_update).
        If successful, status_message contains training summary,
        success_flag is True, and visibility_update hides the progress message.
        On error, returns (error_message, False, visibility_update).

    Examples
    --------
    >>> msg, success, update = train_model(64, 0.001, 0.0, 20, 4)
    >>> if success:
    ...     print("Training completed successfully")
    """
    global trained_model_path
    
    try:
        # Create arguments
        args = argparse.Namespace(
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            epochs=int(epochs),
        )
        
        # Train the model
        train.main(args)
                
        success_msg = f"""
        ### ✅ Training Complete!
        
        **Training Configuration:**
        - **Batch Size**: {batch_size}
        - **Learning Rate**: {learning_rate}
        - **Weight Decay**: {weight_decay}
        - **Epochs**: {epochs}
        - **Workers**: {num_workers}
        
        **Model saved**
        
        🎉 Model is ready for predictions!
        """
        
        return success_msg, True, gr.update(visible=False)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"""
        ### ❌ Training Error
        
        **Error**: {str(e)}
        
        **Details**:
        ```
        {error_details}
        ```
        """
        return error_msg, False, gr.update(visible=False)

def make_predictions(data_dir, models_dir, output_path, target_col, progress=gr.Progress()):
    """
    Generate predictions using the trained model on test data.

    Loads the trained model, makes predictions on the test dataset,
    and saves results to CSV.

    Parameters
    ----------
    data_dir : str
        Path to the processed data directory containing test data.
    models_dir : str
        Path to the directory containing the trained model.
    output_path : str
        Path where prediction results CSV will be saved.
    target_col : str
        Name of the target column in the dataset.
    progress : gr.Progress, optional
        Gradio progress tracker for UI updates, by default gr.Progress().

    Returns
    -------
    tuple of (str, pandas.DataFrame or None, str or None)
        Returns (status_message, predictions_preview, output_file_path).
        If successful, status_message contains summary statistics,
        predictions_preview shows first 20 rows, and output_file_path
        contains the path to saved CSV.
        On error, returns (error_message, None, None).

    Examples
    --------
    >>> stats, preview_df, output_file = make_predictions(
    ...     "data/processed", "models", "models/predictions.csv", "House_Price"
    ... )
    >>> if preview_df is not None:
    ...     print(preview_df.head())
    """
    try:
        progress(0, desc="📂 Loading model...")
        
        # Run predictions
        progress(0.5, desc="🔮 Making predictions...")
        output_csv = predict.run_predict(
            data_dir=data_dir,
            models_dir=models_dir,
            output_path=output_path,
            device="auto",
            target_col=target_col,
        )
        
        progress(0.8, desc="📊 Loading results...")
        
        # Load and display predictions
        df_preds = pd.read_csv(output_csv)
        
        progress(1.0, desc="✅ Complete!")
        
        stats = f"""
        ### ✅ Predictions Complete!
        
        - **Output File**: `{output_csv}`
        - **Total Predictions**: {len(df_preds)}
        - **Columns**: {', '.join(df_preds.columns.tolist())}
        
        📊 Preview shown below
        """
        
        return stats, df_preds.head(20), output_csv
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""
        ### ❌ Prediction Error
        
        **Error**: {str(e)}
        
        **Details**:
        ```
        {error_details}
        ```
        """, None, None

def load_and_preview_data(file_path):
    """
    Load and preview a dataset from CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to load.

    Returns
    -------
    tuple of (str, pandas.DataFrame or None)
        Returns (statistics_summary, data_preview).
        If successful, statistics_summary contains dataset info and
        data_preview shows first 10 rows.
        On error, returns (error_message, None).

    Examples
    --------
    >>> stats, preview = load_and_preview_data("data/raw/dataset.csv")
    >>> if preview is not None:
    ...     print(stats)
    ...     print(preview.head())
    """
    try:
        df = pd.read_csv(file_path)
        
        stats = f"""
        ### 📋 Dataset Preview
        
        - **Total Rows**: {len(df)}
        - **Total Columns**: {len(df.columns)}
        - **Columns**: {', '.join(df.columns.tolist())}
        - **Missing Values**: {df.isnull().sum().sum()}
        """
        
        return stats, df.head(10)
        
    except Exception as e:
        return f"### ❌ Error: {str(e)}", None

def create_training_tabs():
    """
    Create the Training Pipeline tabs for Gradio interface.

    This is the main export function that builds the complete interactive
    training interface with three main tabs: Data Preparation, Model Training,
    and Predictions. Each tab provides controls and visualizations for its
    respective pipeline stage.

    The function creates:
    - Data Preparation tab with split ratio controls and visualizations
    - Model Training tab with hyperparameter configuration
    - Predictions tab for generating and viewing model outputs

    Returns
    -------
    None
        Creates Gradio interface components as side effect.

    Notes
    -----
    This function uses Gradio State objects to manage the pipeline workflow,
    ensuring that training can only occur after data preparation, and
    predictions can only be generated after training.

    Examples
    --------
    >>> with gr.Blocks() as demo:
    ...     create_training_tabs()
    >>> demo.launch()
    """
    
    train_btn_enable = gr.State(value=False)
    predict_btn_enable = gr.State(value=False)
    data_pkg_path = importlib.resources.files("data")
    
    # Build the path to the raw dataset
    full_path_raw = data_pkg_path / "raw" / "house_price_regression_dataset.csv"

    # Tab 1: Data Preparation
    with gr.Tab("📁 Data Preparation"):
        gr.Markdown(
            """
            ### Step 1: Load and Prepare Your Dataset
            
            Provide the path to your raw dataset CSV file and configure the train/val/test split.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                data_path_input = gr.Textbox(
                    value=str(full_path_raw),
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
            outputs=[preview_stats, preview_df]
        )
    
    # Tab 2: Model Training
    with gr.Tab("🎯 Model Training"):
        gr.Markdown(
            """
            ### Step 2: Train the Regression Model
            
            Configure training parameters and start training.
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
                # Loading spinner/message
                training_status = gr.Markdown(visible=False)
                train_output = gr.Markdown()
        
        # Link train button to preparation state
        train_btn_enable.change(
            lambda x: gr.update(interactive=x),
            inputs=[train_btn_enable],
            outputs=[train_start_btn]
        )
        
        # Two-step process: show message, then train
        train_start_btn.click(
            show_training_message,
            inputs=[batch_size_slider, lr_slider, weight_decay_slider, epochs_slider, workers_slider],
            outputs=[training_status, training_status]
        ).then(
            train_model,
            inputs=[batch_size_slider, lr_slider, weight_decay_slider, epochs_slider, workers_slider],
            outputs=[train_output, predict_btn_enable, training_status]
        )
    
    # Tab 3: Predictions
    with gr.Tab("🔮 Make Predictions"):
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