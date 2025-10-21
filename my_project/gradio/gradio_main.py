# /// script
# dependencies = [
#   "gradio",
#   "pandas",
#   "torch",
#   "pytorch-lightning",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn",
#   "numpy",
# ]
# ///

"""
House Price ML Pipeline - Complete Gradio Interface
Combines Data Visualization, Training Pipeline, and Performance Analysis
"""

import gradio as gr
import sys
import os

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))
sys.path.append('./modules')

# Import module functions
from my_project.gradio.Data_Visualization_Interface import create_data_viz_tab
from my_project.gradio.Train_Interface import create_training_tabs
from my_project.gradio.Performance_Analysis_Interface import create_analysis_tab

# Create the main interface
with gr.Blocks(
    title="House Price ML Pipeline - Complete",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        footer {visibility: hidden}
    """
) as demo:
    
    # Header
    gr.Markdown(
        """
        # 🏠 House Price Regression - Complete ML Pipeline
        
        **End-to-end machine learning workflow**: Data Exploration → Training → Prediction → Analysis
        
        Navigate through the tabs to explore each stage of the pipeline.
        
        ---
        """
    )
    
    with gr.Tabs() as main_tabs:
        
        # Tab 1: Data Visualization
        create_data_viz_tab()
        
        # Tab 2: ML Pipeline (Data Prep + Training + Predictions)
        with gr.Tab("🎯 ML Pipeline"):
            gr.Markdown(
                """
                ## Complete Training Pipeline
                
                Follow these steps in order:
                1. **Data Preparation** - Split and prepare your dataset
                2. **Model Training** - Train with custom hyperparameters
                3. **Make Predictions** - Generate predictions on test data
                """
            )
            
            create_training_tabs()
        
        # Tab 3: Performance Analysis
        create_analysis_tab()
    
    # Footer
    gr.Markdown(
        """
        ---
        
        ## 📚 Pipeline Overview
        
        ### 1. 📊 Data Visualization
        - **Distributions**: Explore feature histograms with KDE
        - **Box Plots**: Detect outliers and understand spread
        - **Violin Plots**: Analyze categorical relationships
        - **Correlations**: Hierarchical clustered heatmap
        - **Regression**: Linear and polynomial fits
        - **Filter Explorer**: Interactive data filtering
        
        ### 2. 🎯 ML Pipeline
        - **Data Preparation**: 
          - Custom train/val/test split ratios
          - Automatic data preprocessing
          - Visual split confirmation
        - **Model Training**: 
          - Configurable hyperparameters (batch size, LR, weight decay, epochs)
          - Progress tracking
          - Model checkpointing
        - **Predictions**: 
          - Automated inference on test set
          - CSV export of predictions
        
        ### 3. 📊 Performance Analysis
        - **Metrics Overview**: RMSE, MAE, R², MAPE
        - **Predictions Plot**: Actual vs predicted scatter
        - **Training Curves**: Loss evolution over epochs
        - **Residuals Analysis**: Distribution and Q-Q plot
        - **Worst Predictions**: Top-N error analysis
        - **Error by Quantiles**: Performance across value ranges
        - **Error vs Features**: Correlation analysis
        
        ---
        
        ### 💡 Quick Start Guide
        
        **First Time User?** Follow this workflow:
        
        1. **Explore Data** (Tab 1):
           - Start with "Plot All Distributions" to see all features
           - Check correlations to understand relationships
           - Use regression plots to analyze feature importance
        
        2. **Train Model** (Tab 2):
           - Prepare data with default 60/20/20 split (or customize)
           - Configure training (default: 64 batch, 1e-3 LR, 20 epochs)
           - Wait for training to complete
           - Generate predictions
        
        3. **Analyze Results** (Tab 3):
           - Calculate metrics to see overall performance
           - Plot predictions vs actual
           - Analyze worst predictions to find patterns
           - Check residuals for model assumptions
        
        """
    )
def main():
    demo.launch(
        share=True,
        show_error=True,
        show_api=False
        )
# Launch the application
if __name__ == "__main__":
    main()