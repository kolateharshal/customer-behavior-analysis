import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics
from typing import Dict, Tuple, Any, List

class ScratchLinearRegression:
    """
    Ordinary Least Squares Linear Regression implemented from scratch
    using the Normal Equation: theta = (X^T * X)^(-1) * X^T * y.
    """
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coefficients_ = None
        self.intercept_ = None
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Add column of ones for intercept if required
        if self.fit_intercept:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
        else:
            X_b = X
            
        # Normal Equation: theta = pinv(X_b^T * X_b) * X_b^T * y
        # We use pinv (pseudo-inverse) to avoid singular matrix (non-invertible) issues
        self.theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        if self.fit_intercept:
            self.intercept_ = self.theta[0]
            self.coefficients_ = self.theta[1:]
        else:
            self.intercept_ = 0.0
            self.coefficients_ = self.theta
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            return X.dot(self.coefficients_) + self.intercept_
        else:
            return X.dot(self.coefficients_)


def train_sklearn_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Trains a standard scikit-learn OLS Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge_regression(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Ridge:
    """Trains a Ridge (L2 regularized) Regression model."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_lasso_regression(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Lasso:
    """Trains a Lasso (L1 regularized) Regression model."""
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Computes key performance metrics (MAE, MSE, RMSE, R2, Adjusted R2)
    for a trained regression model.
    """
    predictions = model.predict(X)
    
    mae = metrics.mean_absolute_error(y, predictions)
    mse = metrics.mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y, predictions)
    
    # Adjusted R2 formula
    n = X.shape[0]
    p = X.shape[1]
    
    if n - p - 1 > 0:
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adjusted_r2 = r2
        
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Adjusted_R2": float(adjusted_r2)
    }


def get_model_coefficients(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extracts, maps, and sorts coefficients to show features importances.
    Works for both scikit-learn regression models and custom ScratchLinearRegression.
    """
    if hasattr(model, "coef_"):
        coefs = model.coef_
    elif hasattr(model, "coefficients_"):
        coefs = model.coefficients_
    else:
        raise AttributeError("Model does not have coefficients attribute.")
        
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "Abs_Coefficient": np.abs(coefs)
    })
    
    return coef_df.sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)
