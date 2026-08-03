import os
import sys
import pandas as pd
import numpy as np

# Ensure src folder is in the Python search path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.data_loader import download_and_load_data
from src.preprocessing import preprocess_data
from src.model import (
    ScratchLinearRegression,
    train_sklearn_linear_regression,
    train_ridge_regression,
    train_lasso_regression,
    evaluate_model,
    get_model_coefficients
)
from src.visualizer import (
    plot_correlation_heatmap,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_coefficients
)

def main():
    print("=" * 70)
    print(" HOUSE PRICE PREDICTION - MULTI-MODEL REGRESSION PIPELINE ")
    print("=" * 70)
    
    # 1. Paths and URLs
    DATA_URL = "https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv"
    data_dir = os.path.join(current_dir, "data")
    save_path = os.path.join(data_dir, "kc_house_data.csv")
    viz_dir = os.path.join(current_dir, "visualizations")
    
    # 2. Download and Load Data
    print("\n[Step 1] Loading Dataset...")
    df = download_and_load_data(DATA_URL, save_path)
    
    # 3. Preprocess and Engineer Features
    print("\n[Step 2] Preprocessing Data and Feature Engineering...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # 4. Exploratory Visualization
    print("\n[Step 3] Generating Exploratory Visualizations...")
    plot_correlation_heatmap(df, viz_dir)
    
    # 5. Model Training
    print("\n[Step 4] Training Regression Models...")
    
    # Custom OLS Scratch implementation
    print("  -> Training Custom OLS Regression (From Scratch using Normal Equation)...")
    scratch_model = ScratchLinearRegression()
    scratch_model.fit(X_train, y_train)
    
    # Scikit-learn OLS Linear Regression
    print("  -> Training Scikit-Learn OLS Linear Regression...")
    sklearn_lr = train_sklearn_linear_regression(X_train, y_train)
    
    # Ridge L2 Regularization (helps with collinearity in coordinates/engineered columns)
    print("  -> Training Scikit-Learn Ridge Regression (alpha=10.0)...")
    sklearn_ridge = train_ridge_regression(X_train, y_train, alpha=10.0)
    
    # Lasso L1 Regularization (performs feature selection by forcing small weights to 0)
    print("  -> Training Scikit-Learn Lasso Regression (alpha=100.0)...")
    sklearn_lasso = train_lasso_regression(X_train, y_train, alpha=100.0)
    
    # 6. Evaluation
    print("\n[Step 5] Evaluating Models on Test Set...")
    metrics_scratch = evaluate_model(scratch_model, X_test, y_test)
    metrics_lr = evaluate_model(sklearn_lr, X_test, y_test)
    metrics_ridge = evaluate_model(sklearn_ridge, X_test, y_test)
    metrics_lasso = evaluate_model(sklearn_lasso, X_test, y_test)
    
    # Assemble results table
    results_df = pd.DataFrame({
        "Model": ["Scratch OLS", "Sklearn OLS", "Ridge (L2)", "Lasso (L1)"],
        "MAE ($)": [metrics_scratch["MAE"], metrics_lr["MAE"], metrics_ridge["MAE"], metrics_lasso["MAE"]],
        "RMSE ($)": [metrics_scratch["RMSE"], metrics_lr["RMSE"], metrics_ridge["RMSE"], metrics_lasso["RMSE"]],
        "R2 Score": [metrics_scratch["R2"], metrics_lr["R2"], metrics_ridge["R2"], metrics_lasso["R2"]],
        "Adjusted R2": [metrics_scratch["Adjusted_R2"], metrics_lr["Adjusted_R2"], metrics_ridge["Adjusted_R2"], metrics_lasso["Adjusted_R2"]]
    })
    
    # Display formatted output
    results_display = results_df.copy()
    results_display["MAE ($)"] = results_display["MAE ($)"].apply(lambda x: f"${x:,.2f}")
    results_display["RMSE ($)"] = results_display["RMSE ($)"].apply(lambda x: f"${x:,.2f}")
    results_display["R2 Score"] = results_display["R2 Score"].round(5)
    results_display["Adjusted R2"] = results_display["Adjusted R2"].round(5)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON REPORT".center(80))
    print("="*80)
    print(results_display.to_string(index=False))
    print("="*80)
    
    # Check if custom from scratch implementation matches sklearn OLS (they should match exactly)
    diff_r2 = abs(metrics_scratch["R2"] - metrics_lr["R2"])
    if diff_r2 < 1e-7:
        print("  * Verification: Scratch OLS and Sklearn OLS metrics match perfectly! (Diff < 1e-7)")
    else:
        print(f"  * Warning: Slight numeric difference between Scratch and Sklearn OLS (Diff: {diff_r2:.2e})")
        
    # 7. Feature Coefficient Analysis
    print("\n[Step 6] Extracting Feature Coefficients...")
    coef_df = get_model_coefficients(sklearn_lr, feature_names)
    print("\nTop 15 Most Influential Features (Sklearn OLS):")
    print(coef_df.head(15).to_string(index=False))
    
    # 8. Generate Visualizations (based on standard Sklearn OLS)
    print("\n[Step 7] Generating Model Diagnostic Plots...")
    y_pred = sklearn_lr.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred, viz_dir, "Linear Regression")
    plot_residuals(y_test, y_pred, viz_dir, "Linear Regression")
    plot_coefficients(coef_df, viz_dir, "Linear Regression", top_n=15)
    
    print("\n" + "="*70)
    print(" PIPELINE EXECUTION COMPLETE ")
    print("="*70)
    print(f"Visualizations saved to: {viz_dir}")
    print(f"Dataset cached in: {save_path}\n")

if __name__ == "__main__":
    main()
