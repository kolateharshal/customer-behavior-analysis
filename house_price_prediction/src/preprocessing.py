import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Cleans the dataset, performs feature engineering, one-hot encodes zipcode,
    splits into train/test sets, scales features, and returns components.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The raw house sales data.
        
    Returns:
    --------
    X_train : np.ndarray
        Standardized training features.
    X_test : np.ndarray
        Standardized testing features.
    y_train : np.ndarray
        Training target (price).
    y_test : np.ndarray
        Testing target (price).
    scaler : StandardScaler
        The fitted StandardScaler object for numerical features.
        Used to inverse transform if needed.
    feature_names : List[str]
        List of feature column names corresponding to columns of X.
    """
    # 1. Clean Data
    df_clean = df.copy()
    
    # Drop identifier
    if 'id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['id'])
        
    # Remove records with anomaly / noise: 0 bedrooms or 0 bathrooms
    initial_rows = len(df_clean)
    df_clean = df_clean[(df_clean['bedrooms'] > 0) & (df_clean['bathrooms'] > 0)]
    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with 0 bedrooms or bathrooms.")
        
    # 2. Parse Date
    # Date format in dataset is typically "YYYYMMDDT000000" (e.g., "20141013T000000")
    if 'date' in df_clean.columns:
        df_clean['sale_year'] = df_clean['date'].str[:4].astype(int)
        df_clean['sale_month'] = df_clean['date'].str[4:6].astype(int)
        df_clean = df_clean.drop(columns=['date'])
    else:
        # Fallbacks if date is not in standard string format
        df_clean['sale_year'] = 2015
        df_clean['sale_month'] = 6
        
    # 3. Feature Engineering
    # House Age at sale time
    df_clean['house_age'] = df_clean['sale_year'] - df_clean['yr_built']
    # Adjust for cases where sale_year < yr_built (data recording issue)
    df_clean['house_age'] = df_clean['house_age'].clip(lower=0)
    
    # Renovated flag
    df_clean['is_renovated'] = (df_clean['yr_renovated'] > 0).astype(int)
    
    # Years since renovation
    df_clean['years_since_renovation'] = df_clean.apply(
        lambda row: row['sale_year'] - row['yr_renovated'] if row['yr_renovated'] > 0 else row['house_age'],
        axis=1
    )
    df_clean['years_since_renovation'] = df_clean['years_since_renovation'].clip(lower=0)
    
    # Average space per room
    df_clean['sqft_per_room'] = df_clean['sqft_living'] / (df_clean['bedrooms'] + df_clean['bathrooms'])
    
    # Has a basement flag
    df_clean['has_basement'] = (df_clean['sqft_basement'] > 0).astype(int)
    
    # Ratio of sqft_above to total sqft_living
    df_clean['sqft_above_ratio'] = df_clean['sqft_above'] / (df_clean['sqft_living'] + 1e-5)
    
    # 4. Handle Categorical Columns
    # zipcode is categorical but represented as integers. Location is key for real estate.
    # One-hot encode zipcodes to capture geographical premium.
    df_clean = pd.get_dummies(df_clean, columns=['zipcode'], drop_first=True)
    
    # 5. Define target and features
    y = df_clean['price'].values
    X_df = df_clean.drop(columns=['price'])
    
    feature_names = list(X_df.columns)
    
    # 6. Train-Test Split (80/20 split)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # 7. Scale Numerical Features
    # Identify numerical columns (exclude binary dummy columns like is_renovated, has_basement, and zipcode dummies)
    columns_to_scale = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
        'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
        'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
        'sqft_living15', 'sqft_lot15', 'sale_year', 'sale_month', 
        'house_age', 'years_since_renovation', 'sqft_per_room', 'sqft_above_ratio'
    ]
    # Filter to make sure columns exist in current dataframe
    columns_to_scale = [col for col in columns_to_scale if col in X_train_raw.columns]
    
    # Initialize Scaler
    scaler = StandardScaler()
    
    # We fit the scaler only on X_train_raw's numerical columns to prevent leakage
    X_train_scaled = X_train_raw.copy()
    X_test_scaled = X_test_raw.copy()
    
    # Cast to float for scaling
    X_train_scaled[columns_to_scale] = X_train_scaled[columns_to_scale].astype(float)
    X_test_scaled[columns_to_scale] = X_test_scaled[columns_to_scale].astype(float)
    
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train_scaled[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test_scaled[columns_to_scale])
    
    # Convert back to numpy arrays for sklearn input
    # Note: Pandas get_dummies can generate bool columns for dummies. Convert those to int/float.
    X_train = X_train_scaled.values.astype(float)
    X_test = X_test_scaled.values.astype(float)
    
    print(f"Preprocessed features shape: {X_train.shape}")
    print(f"Number of target columns: {len(y_train)}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names

if __name__ == "__main__":
    # Test stub
    import os
    from data_loader import download_and_load_data
    
    DATA_URL = "https://raw.githubusercontent.com/dhyan6/data-science-projects/main/kc_house_data.csv"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(os.path.dirname(current_dir), "data", "kc_house_data.csv")
    
    try:
        raw_df = download_and_load_data(DATA_URL, SAVE_PATH)
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(raw_df)
        print("Success! Feature list sample:", feature_names[:10])
    except Exception as e:
        print(f"Error testing preprocessing: {e}")
