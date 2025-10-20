"""
Data Visualization Module for Gradio Interface
Interactive visualizations for house price dataset exploration
"""

import gradio as gr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import importlib.resources
import os

# Load the dataset
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")
# --- ESTO NO ES RECOMENDABLE ---

try:
    # 1. Encontrar la ruta al paquete "data" instalado en site-packages
    data_pkg_path = importlib.resources.files("data")
    
    # 2. Construir la ruta relativa DENTRO de ese paquete
    full_path = data_pkg_path / "raw" / "house_price_regression_dataset.csv"
    
    # 3. Abrir el archivo
    df = pd.read_csv(full_path)

except ModuleNotFoundError:
    print("Error: El 'paquete' data no está instalado.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en {full_path}")
# --- NO HAGAS ESTO ---
# df = pd.read_csv('data/raw/house_price_regression_dataset.csv')
X = df.drop(columns=['House_Price'])
y = df['House_Price']

# Get feature names
features = df.columns.drop('House_Price').tolist()
categorical_vars = ["Num_Bedrooms", "Num_Bathrooms", "Neighborhood_Quality", "Garage_Size"]

# Color options
color_options = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
palette_options = ["pastel", "deep", "muted", "bright", "dark", "colorblind", "Set2", "Set3", "husl"]
cmap_options = ["coolwarm", "viridis", "plasma", "RdYlBu", "RdYlGn", "Spectral", "seismic"]

def plot_feature_distribution(feature, bins, kde, color, alpha):
    """Plot histogram with KDE for a selected feature"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    
    if feature == "House_Price":
        sns.histplot(y, color=color, kde=kde, bins=bins, ax=ax, alpha=alpha)
    else:
        sns.histplot(X[feature].dropna(), color=color, kde=kde, bins=bins, ax=ax, alpha=alpha)
    
    ax.set_title(f"{feature} Distribution", fontsize=16, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def plot_all_distributions():
    """Plot all feature distributions in a grid"""
    fig = plt.figure(figsize=(20, 25))
    sns.set_theme(style="darkgrid")
    gs = plt.GridSpec(3, 3, figure=fig)

    for i, feature in enumerate(features):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        sns.histplot(X[feature].dropna(), color="orange", kde=True, ax=ax)
        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
    
    axe = fig.add_subplot(gs[2, 1:])
    sns.histplot(y, color="blue", kde=True, bins=30, ax=axe)
    axe.set_title(f"House Price Distribution", fontsize=16, fontweight='bold')
    axe.set_xlabel("House Price")
    axe.set_ylabel("Frequency")

    fig.suptitle('Feature Distributions\n\n\n', fontsize=25)
    plt.tight_layout()
    return fig

def plot_violin(variable, palette, split_violin):
    """Plot violin plot for categorical variables vs House Price"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if split_violin and variable in ["Num_Bedrooms", "Num_Bathrooms"]:
        df_temp = df.copy()
        median_val = df_temp[variable].median()
        df_temp['split'] = df_temp[variable] <= median_val
        
        sns.violinplot(
            data=df_temp,
            x=variable,
            y=y,
            ax=ax,
            inner="box",
            cut=0,
            hue='split',
            palette=palette,
            legend=True,
            split=False
        )
    else:
        sns.violinplot(
            data=df,
            x=variable,
            y=y,
            ax=ax,
            inner="box",
            cut=0,
            hue=variable,
            palette=palette,
            legend=False
        )
    
    ax.set_title(f"House Price vs {variable}", fontsize=14, fontweight='bold')
    ax.set_xlabel(variable)
    ax.set_ylabel("House Price")
    plt.tight_layout()
    return fig

def plot_all_violins():
    """Plot all violin plots in a grid"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, var in enumerate(categorical_vars):
        sns.violinplot(
            data=df,
            x=var,
            y=y,
            ax=axes[i],
            inner="box",
            cut=0,
            hue=var,
            palette="pastel",
            legend=False
        )
        axes[i].set_title(f"House Price vs {var}", fontsize=12)
        axes[i].set_xlabel(var)
        axes[i].set_ylabel("House Price")

    plt.tight_layout()
    return fig

def plot_correlation_heatmap(annot, cmap, fmt):
    """Plot correlation heatmap with hierarchical clustering"""
    corr_matrix = df.corr()
    
    # Use clustermap to get ordering
    cg = sns.clustermap(corr_matrix, cmap=cmap, linewidths=0.5, figsize=(10, 8))
    ordered_indices = cg.dendrogram_row.reordered_ind
    plt.close()
    
    # Reorder the correlation matrix
    corr_matrix = corr_matrix.iloc[ordered_indices, ordered_indices]
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_theme(style="white")
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=fmt, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap (Hierarchically Clustered)", fontsize=14)
    plt.tight_layout()
    return fig

def plot_regression(x_feature, order, scatter_alpha, line_color, ci):
    """Plot regression plot between selected feature and House Price"""
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.regplot(
        data=df, 
        x=x_feature, 
        y="House_Price", 
        scatter_kws={"alpha": scatter_alpha},
        order=order,
        line_kws={"color": line_color},
        ci=ci if ci > 0 else None,
        ax=ax
    )
    ax.set_title(f"House Price vs {x_feature} (Polynomial Order: {order})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_boxplot(variable, show_outliers, palette):
    """Plot boxplot for a feature"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if variable in categorical_vars:
        sns.boxplot(
            data=df, 
            x=variable, 
            y="House_Price", 
            hue=variable, 
            palette=palette, 
            ax=ax, 
            legend=False,
            showfliers=show_outliers
        )
        ax.set_title(f"House Price Distribution by {variable}", fontsize=14, fontweight='bold')
    else:
        sns.boxplot(
            data=df, 
            y=variable, 
            ax=ax, 
            color="skyblue",
            showfliers=show_outliers
        )
        ax.set_title(f"{variable} Distribution (Boxplot)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def filter_data(min_price, max_price, min_sqft, max_sqft):
    """Filter dataset and return statistics"""
    filtered = df[
        (df['House_Price'] >= min_price) & 
        (df['House_Price'] <= max_price) &
        (df['Square_Footage'] >= min_sqft) &
        (df['Square_Footage'] <= max_sqft)
    ]
    
    stats = f"""
    ### 📊 Filtered Data Statistics
    - **Total Records**: {len(filtered)} / {len(df)} ({len(filtered)/len(df)*100:.1f}%)
    - **Average Price**: ${filtered['House_Price'].mean():,.2f}
    - **Average Square Footage**: {filtered['Square_Footage'].mean():,.0f} sq ft
    - **Price Range**: ${filtered['House_Price'].min():,.0f} - ${filtered['House_Price'].max():,.0f}
    """
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        filtered['Square_Footage'], 
        filtered['House_Price'], 
        alpha=0.5, 
        c=filtered['House_Price'],
        cmap='viridis',
        s=50
    )
    ax.set_xlabel('Square Footage', fontsize=12)
    ax.set_ylabel('House Price', fontsize=12)
    ax.set_title(f'Filtered Data: Price vs Square Footage ({len(filtered)} houses)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='House Price')
    plt.tight_layout()
    
    return stats, fig

def create_data_viz_tab():
    """Create the Data Visualization tab - Main export function"""
    
    with gr.Tab("📊 Data Visualization"):
        gr.Markdown(
            """
            ## Interactive Data Visualization
            
            Explore the house pricing dataset through various interactive visualizations.
            """
        )
        
        with gr.Tabs():
            # Feature Distributions
            with gr.Tab("📊 Distributions"):
                gr.Markdown("### Explore individual feature distributions")
                with gr.Row():
                    with gr.Column(scale=1):
                        feature_dropdown = gr.Dropdown(
                            choices=features + ["House_Price"],
                            value="House_Price",
                            label="Select Feature"
                        )
                        bins_slider = gr.Slider(5, 100, 30, step=5, label="Number of Bins")
                        kde_checkbox = gr.Checkbox(value=True, label="Show KDE")
                        color_dropdown = gr.Dropdown(choices=color_options, value="blue", label="Color")
                        alpha_slider = gr.Slider(0.1, 1.0, 0.7, step=0.1, label="Transparency")
                        single_dist_btn = gr.Button("Plot Distribution", variant="primary")
                        all_dist_btn = gr.Button("Plot All Distributions")
                    with gr.Column(scale=2):
                        dist_plot = gr.Plot(label="Distribution Plot")
                
                single_dist_btn.click(
                    plot_feature_distribution, 
                    inputs=[feature_dropdown, bins_slider, kde_checkbox, color_dropdown, alpha_slider], 
                    outputs=dist_plot
                )
                all_dist_btn.click(plot_all_distributions, outputs=dist_plot)

            # Box Plots
            with gr.Tab("📦 Box Plots"):
                gr.Markdown("### Box plots for outlier detection")
                with gr.Row():
                    with gr.Column(scale=1):
                        box_dropdown = gr.Dropdown(choices=features + ["House_Price"], value="House_Price", label="Select Feature")
                        outliers_checkbox = gr.Checkbox(value=True, label="Show Outliers")
                        box_palette = gr.Dropdown(choices=palette_options, value="Set2", label="Color Palette")
                        box_btn = gr.Button("Plot Boxplot", variant="primary")
                    with gr.Column(scale=2):
                        box_plot = gr.Plot(label="Box Plot")
                
                box_btn.click(plot_boxplot, inputs=[box_dropdown, outliers_checkbox, box_palette], outputs=box_plot)
            
            # Violin Plots
            with gr.Tab("🎻 Violin Plots"):
                gr.Markdown("### Categorical features vs House Price")
                with gr.Row():
                    with gr.Column(scale=1):
                        cat_dropdown = gr.Dropdown(choices=categorical_vars, value="Num_Bedrooms", label="Categorical Variable")
                        violin_palette = gr.Dropdown(choices=palette_options, value="pastel", label="Color Palette")
                        split_checkbox = gr.Checkbox(value=False, label="Comparison Split")
                        single_violin_btn = gr.Button("Plot Violin", variant="primary")
                        all_violin_btn = gr.Button("Plot All Violins")
                    with gr.Column(scale=2):
                        violin_plot = gr.Plot(label="Violin Plot")
                
                single_violin_btn.click(plot_violin, inputs=[cat_dropdown, violin_palette, split_checkbox], outputs=violin_plot)
                all_violin_btn.click(plot_all_violins, outputs=violin_plot)
            
            # Correlation Heatmap
            with gr.Tab("🔥 Correlations"):
                gr.Markdown("### Correlation matrix with hierarchical clustering")
                with gr.Row():
                    with gr.Column(scale=1):
                        annot_checkbox = gr.Checkbox(value=True, label="Show Values")
                        cmap_dropdown = gr.Dropdown(choices=cmap_options, value="coolwarm", label="Color Map")
                        fmt_dropdown = gr.Dropdown(choices=[".2f", ".1f", ".3f"], value=".2f", label="Format")
                        heatmap_btn = gr.Button("Generate Heatmap", variant="primary")
                    with gr.Column(scale=2):
                        heatmap_plot = gr.Plot(label="Correlation Heatmap")
                
                heatmap_btn.click(plot_correlation_heatmap, inputs=[annot_checkbox, cmap_dropdown, fmt_dropdown], outputs=heatmap_plot)
            
            # Regression Analysis
            with gr.Tab("📈 Regression"):
                gr.Markdown("### Linear/Polynomial regression analysis")
                with gr.Row():
                    with gr.Column(scale=1):
                        reg_dropdown = gr.Dropdown(choices=features, value="Square_Footage", label="Feature")
                        order_slider = gr.Slider(1, 5, 1, step=1, label="Polynomial Order")
                        scatter_alpha_slider = gr.Slider(0.1, 1.0, 0.5, step=0.1, label="Point Transparency")
                        line_color = gr.Dropdown(choices=color_options, value="red", label="Line Color")
                        ci_slider = gr.Slider(0, 99, 95, step=5, label="Confidence Interval (%)")
                        reg_btn = gr.Button("Plot Regression", variant="primary")
                    with gr.Column(scale=2):
                        reg_plot = gr.Plot(label="Regression Plot")
                
                reg_btn.click(plot_regression, inputs=[reg_dropdown, order_slider, scatter_alpha_slider, line_color, ci_slider], outputs=reg_plot)
            
            # Data Filter
            with gr.Tab("🔍 Filter Explorer"):
                gr.Markdown("### Filter data by price and square footage")
                with gr.Row():
                    with gr.Column(scale=1):
                        price_range = gr.Slider(
                            minimum=float(df['House_Price'].min()),
                            maximum=float(df['House_Price'].max()),
                            value=float(df['House_Price'].min()),
                            label="Min House Price"
                        )
                        price_range_max = gr.Slider(
                            minimum=float(df['House_Price'].min()),
                            maximum=float(df['House_Price'].max()),
                            value=float(df['House_Price'].max()),
                            label="Max House Price"
                        )
                        sqft_range = gr.Slider(
                            minimum=float(df['Square_Footage'].min()),
                            maximum=float(df['Square_Footage'].max()),
                            value=float(df['Square_Footage'].min()),
                            label="Min Square Footage"
                        )
                        sqft_range_max = gr.Slider(
                            minimum=float(df['Square_Footage'].min()),
                            maximum=float(df['Square_Footage'].max()),
                            value=float(df['Square_Footage'].max()),
                            label="Max Square Footage"
                        )
                        filter_btn = gr.Button("Apply Filter", variant="primary")
                    with gr.Column(scale=2):
                        filter_stats = gr.Markdown()
                        filter_plot = gr.Plot(label="Filtered Data")
                
                filter_btn.click(filter_data, inputs=[price_range, price_range_max, sqft_range, sqft_range_max], outputs=[filter_stats, filter_plot])
        
        gr.Markdown(
            """
            ---
            **Tip**: Adjust sliders and parameters to customize visualizations!
            """
        )