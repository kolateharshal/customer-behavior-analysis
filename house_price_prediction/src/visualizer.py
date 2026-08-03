import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List

# Setup visual defaults
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Plots and saves a correlation heatmap for numerical features.
    Excludes high-dimensional dummy columns to keep the heatmap readable.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_cols = [
        'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
        'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
        'sqft_basement', 'house_age', 'years_since_renovation', 
        'sqft_per_room', 'sqft_above_ratio'
    ]
    # Filter to make sure they exist in the dataframe
    cols_to_use = [col for col in num_cols if col in df.columns]
    
    plt.figure(figsize=(14, 11))
    corr = df[cols_to_use].corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap='coolwarm', 
        annot=True, 
        fmt=".2f", 
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10}
    )
    plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()
    print("Saved correlation_matrix.png")


def plot_actual_vs_predicted(y_actual: np.ndarray, y_pred: np.ndarray, output_dir: str, model_name: str = "Linear Regression"):
    """
    Plots and saves an Actual vs Predicted prices scatter plot with a y=x reference line.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    # Use alpha to handle dense scatter plots nicely
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.3, color='#1f77b4', edgecolor='none')
    
    # Plot diagonal line
    max_val = max(y_actual.max(), y_pred.max())
    min_val = min(y_actual.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
    plt.title(f'Actual vs. Predicted House Prices ({model_name})', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Format axes labels with dollar sign and commas
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, pos: f"${int(x):,}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    file_name = f'actual_vs_predicted_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(os.path.join(output_dir, file_name), dpi=300)
    plt.close()
    print(f"Saved {file_name}")


def plot_residuals(y_actual: np.ndarray, y_pred: np.ndarray, output_dir: str, model_name: str = "Linear Regression"):
    """
    Plots and saves two residual analysis plots: 
    1. Residual vs. Fitted (Fitted Values vs Residuals)
    2. Distribution of Residuals (Histogram with KDE)
    """
    os.makedirs(output_dir, exist_ok=True)
    residuals = y_actual - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Residual Scatter Plot
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3, color='#2ca02c', ax=axes[0], edgecolor='none')
    axes[0].axhline(y=0, color='#d62728', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Price ($)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
    axes[0].set_title('Residuals vs. Predicted Values', fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, pos: f"${int(x):,}")
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].yaxis.set_major_formatter(formatter)
    
    # 2. Residual Distribution Histogram
    sns.histplot(residuals, kde=True, color='#e377c2', ax=axes[1], bins=50)
    axes[1].set_xlabel('Residual Amount ($)', fontsize=11, fontweight='bold')
    axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].xaxis.set_major_formatter(formatter)
    
    plt.suptitle(f'Residual Analysis ({model_name})', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    file_name = f'residuals_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(os.path.join(output_dir, file_name), dpi=300)
    plt.close()
    print(f"Saved {file_name}")


def plot_coefficients(coef_df: pd.DataFrame, output_dir: str, model_name: str = "Linear Regression", top_n: int = 15):
    """
    Plots and saves feature coefficients. Displays top_n features by effect size.
    Features are color-coded: blue for positive impact on price, red for negative.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select top features by magnitude of the coefficient
    top_coefs = coef_df.head(top_n).copy()
    
    plt.figure(figsize=(12, 8))
    
    # Color coding based on sign
    colors = ['#1f77b4' if x >= 0 else '#d62728' for x in top_coefs['Coefficient']]
    
    sns.barplot(
        x='Coefficient', 
        y='Feature', 
        data=top_coefs, 
        palette=colors,
        hue='Feature',
        legend=False
    )
    
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel('Standardized Coefficient (Standard Deviation Change in Price)', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Coefficients ({model_name})\nBlue: Positive Impact | Red: Negative Impact', 
              fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    file_name = f'coefficients_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(os.path.join(output_dir, file_name), dpi=300)
    plt.close()
    print(f"Saved {file_name}")
