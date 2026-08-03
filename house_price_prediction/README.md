# House Price Prediction Project - Linear Regression Analysis

## 📌 Project Overview & Internship Objectives
This project was developed as part of a data science internship to predict house sales prices in King County, Washington (including Seattle) based on physical characteristics, locations, structural ratings, and chronological factors. 

The primary objective is to implement a robust, production-grade machine learning pipeline utilizing multiple linear regression techniques. We contrast standard **Ordinary Least Squares (OLS)** models (both built from scratch and using `scikit-learn`) with regularized variants (**Ridge L2** and **Lasso L1**) to show how to control overfitting and stabilize models under multicollinearity.

---

## 📂 Project Structure
The workspace is structured cleanly to separate raw data, processing components, models, outputs, and execution pipelines:

```
house_price_prediction/
├── data/
│   └── kc_house_data.csv          # Raw dataset downloaded from GitHub
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Automated downloader & cache manager
│   ├── preprocessing.py           # Cleaning, feature engineering, and scaling
│   ├── model.py                  # Custom OLS from scratch & sklearn model trainers
│   └── visualizer.py             # Seaborn diagnostic plotting suite
├── visualizations/                # Generated analysis and evaluation plots
│   ├── correlation_matrix.png
│   ├── actual_vs_predicted_linear_regression.png
│   ├── residuals_linear_regression.png
│   └── coefficients_linear_regression.png
├── main.py                       # Pipeline orchestrator
├── requirements.txt              # Project dependencies
└── README.md                     # Internship project report (this document)
```

---

## 📊 Dataset & Feature Descriptions
The dataset contains historical transaction data of **21,613 homes** sold between May 2014 and May 2015. 
After removing data anomalies (e.g., residential sales listed with 0 bedrooms or 0 bathrooms), the pipeline processes **17,277 records** across **94 features** (including engineered features and zipcode dummies):

### Core Features Used
*   **Target (Dependent Variable):** `price` - The listing sale price of the house.
*   **Size (Square Footage):**
    *   `sqft_living` / `sqft_living15`: Living space of the home / nearest 15 neighbors.
    *   `sqft_lot` / `sqft_lot15`: Total land parcel size / nearest 15 neighbors.
    *   `sqft_above` / `sqft_basement`: Size splits of space above and below ground level.
*   **Rooms:** `bedrooms`, `bathrooms` (including fractional counts).
*   **Chronological:** `yr_built` (construction year), `yr_renovated` (renovation year, 0 if untouched).
*   **Geographical:** `zipcode` (one-hot encoded for ~70 areas), `lat` (latitude), `long` (longitude).
*   **Quality & Environment:** `grade` (construction quality 1-13), `condition` (maintenance scale 1-5), `waterfront` (sea/lake view dummy), `view` (scenery index 0-4).

### 🛠️ Feature Engineering
To extract richer information, we created several domain-specific engineered features:
1.  **`house_age`**: Year of sale minus `yr_built` (representing depreciation).
2.  **`is_renovated`**: Binary flag indicating if the house has undergone renovations.
3.  **`years_since_renovation`**: Time elapsed since construction or latest renovation.
4.  **`sqft_per_room`**: Space density metric calculated as `sqft_living / (bedrooms + bathrooms)`.
5.  **`has_basement`**: Binary indicator showing if a basement is present.
6.  **`sqft_above_ratio`**: Ratio of above-ground living space to total living area.

---

## 📐 Modeling Methodology
We train and compare four regression models to evaluate numerical stability, predictive accuracy, and feature selection capabilities.

### 1. Custom OLS Regression (From Scratch)
We implemented standard OLS using the **Normal Equation**:
$$\theta = (X^T X)^{-1} X^T y$$
To handle potential rank deficiency (non-invertibility of $X^T X$ due to multicollinearity), we implemented the closed-form solver utilizing NumPy's Moore-Penrose pseudo-inverse (`np.linalg.pinv`):
$$\theta = X^+ y$$
This ensures numerical stability when calculating coefficients without relying on external machine learning packages.

### 2. Scikit-Learn OLS Regression
Serves as the baseline implementation. It solves the OLS problem using scipy's lapack-based solver.

### 3. Ridge Regression (L2 Regularization)
Adds a squared magnitude penalty to the loss function to shrink coefficients towards zero:
$$\text{Loss} = \text{MSE} + \alpha \sum_{j=1}^{P} \theta_j^2$$
We set $\alpha = 10.0$ to penalize extreme weights and handle highly correlated features.

### 4. Lasso Regression (L1 Regularization)
Adds an absolute magnitude penalty to the loss function, forcing non-essential weights to exactly zero:
$$\text{Loss} = \text{MSE} + \alpha \sum_{j=1}^{P} |\theta_j|$$
We set $\alpha = 100.0$ to encourage sparse coefficient matrices, performing automated feature selection.

---

## 📈 Evaluation Results & Comparison

Below is the comparative evaluation of the models on the test set (20% split):

| Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score | Adjusted $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Scratch OLS** | \$93,677.72 | \$161,781.14 | 0.79881 | 0.79434 |
| **Sklearn OLS** | \$93,706.92 | \$161,783.31 | 0.79881 | 0.79433 |
| **Ridge (L2)** | \$93,885.80 | \$162,028.50 | 0.79820 | 0.79371 |
| **Lasso (L1)** | \$93,569.53 | \$162,155.98 | 0.79788 | 0.79338 |

---

## 🔍 Key Insights & Analysis

### 1. Algorithmic Equivalence (Scratch vs. Sklearn)
The **Scratch OLS** and **Sklearn OLS** models returned virtually identical metrics (both hitting an $R^2$ of **0.79881**). The minor difference (RMSE difference of only \$2.17) is due to internal floating-point optimization routines in LAPACK vs. our pseudo-inverse solver. This validates the mathematical correctness of our scratch matrix implementation.

### 2. The Multicollinearity Pitfall
In OLS coefficients, we notice that `yr_built` ($-\$6.88 \times 10^6$) and `house_age` ($-\$6.87 \times 10^6$) have extremely large negative coefficients. This is a classic sign of **perfect multicollinearity**. 
Since `house_age = sale_year - yr_built`, and the sale year is nearly constant (2014 or 2015), the relationship is linear:
$$\text{house\_age} + \text{yr\_built} \approx \text{constant}$$
The OLS optimizer creates massive, opposing coefficients that cancel each other out. Standardizing and training with **Ridge L2 regularization** shrinks these coefficients significantly, proving the utility of regularized models in production networks.

### 3. Location, Location, Location (Geographical Premium)
By one-hot encoding zipcodes, the model reveals that geographical location is the strongest driver of house premiums:
*   **`zipcode_98039` (Medina, WA)**: Standardized coefficient is **+\$1.23M** (the highest positive driver).
*   **`zipcode_98004` (Bellevue, WA)**: Standardized coefficient is **+\$650k**.
*   **`zipcode_98112` (Capitol Hill/Madison Park)**: Standardized coefficient is **+\$540k**.
These results perfectly align with Seattle real estate reality, demonstrating that our regression coefficients reflect real-world asset valuations.

---

## 🖼️ Visualizations Guide
All diagnostic plots are saved in the `visualizations/` directory:

1.  **`correlation_matrix.png`**: Heatmap displaying correlations between numerical variables, illustrating which features (like `grade` and `sqft_living`) have strong positive correlations with `price`.
2.  **`actual_vs_predicted_linear_regression.png`**: A scatter plot comparing predicted prices with actual sale prices. The red diagonal line ($y=x$) marks perfect predictions.
3.  **`residuals_linear_regression.png`**: Diagnostic plots displaying residual errors. The left graph checks for homoscedasticity (constant variance), while the right histogram plots the residual distribution to verify the normality assumption of OLS.
4.  **`coefficients_linear_regression.png`**: Horizontal bar chart summarizing the top 15 features sorted by standardized coefficient magnitude. Blue bars indicate price-boosting features, while red indicates price-reducing features (like age).

---

## 🚀 How to Run the Project
To run the entire pipeline from scratch, follow these instructions:

### 1. Install Dependencies
Make sure you have python 3 installed. Install libraries via `pip`:
```bash
pip install -r requirements.txt
```

### 2. Execute the Pipeline
Run the main script:
```bash
python main.py
```
This will automatically:
- Download the dataset to `data/` (if missing).
- Preprocess data and perform feature engineering.
- Train all models.
- Print the metrics comparison table.
- Print the top feature coefficients.
- Generate and save the visualizations in the `visualizations/` directory.
