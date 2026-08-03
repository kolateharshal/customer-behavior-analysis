import os
import pandas as pd
import numpy as np

# Import pipeline components
from src.preprocessing import preprocess_data
from src.model import train_sklearn_linear_regression

def get_user_input(feature_names, df):
    """Prompts user for house attributes via console input with validation."""
    print("\n" + "="*60)
    print("🏠 ENTER HOUSE DETAILS FOR PRICE PREDICTION")
    print("="*60)
    
    # 1. Zipcode validation
    available_zipcodes = sorted(df['zipcode'].unique())
    while True:
        try:
            zip_input = int(input(f"Enter Zipcode (e.g., {available_zipcodes[0]} to {available_zipcodes[-1]}): ").strip())
            if zip_input in available_zipcodes:
                zipcode = zip_input
                break
            else:
                print(f"⚠️ Zipcode {zip_input} not found in King County dataset. Try another (like 98112, 98039, or 98004).")
        except ValueError:
            print("⚠️ Please enter a valid 5-digit integer zipcode.")
            
    # 2. Bedrooms
    while True:
        try:
            bedrooms = float(input("Number of Bedrooms (e.g., 3): ").strip())
            if bedrooms >= 0:
                break
            print("⚠️ Number of bedrooms cannot be negative.")
        except ValueError:
            print("⚠️ Please enter a valid number.")

    # 3. Bathrooms
    while True:
        try:
            bathrooms = float(input("Number of Bathrooms (e.g., 2.5): ").strip())
            if bathrooms >= 0:
                break
            print("⚠️ Number of bathrooms cannot be negative.")
        except ValueError:
            print("⚠️ Please enter a valid number.")

    # 4. Sqft Living
    while True:
        try:
            sqft_living = float(input("Living Area Size (in sqft, e.g., 2000): ").strip())
            if sqft_living > 0:
                break
            print("⚠️ Living area size must be positive.")
        except ValueError:
            print("⚠️ Please enter a valid number.")

    # 5. Sqft Lot
    while True:
        try:
            sqft_lot = float(input("Total Land Lot Size (in sqft, e.g., 5000): ").strip())
            if sqft_lot > 0:
                break
            print("⚠️ Lot size must be positive.")
        except ValueError:
            print("⚠️ Please enter a valid number.")

    # 6. Year Built
    while True:
        try:
            yr_built = int(input("Year Built (e.g., 1995): ").strip())
            if 1800 <= yr_built <= 2026:
                break
            print("⚠️ Please enter a year between 1800 and 2026.")
        except ValueError:
            print("⚠️ Please enter a valid year.")

    # 7. Quality Grade (Standard 1-13 scale)
    while True:
        grade_input = input("Construction Quality Grade (1-13, default is 7 [Average]): ").strip()
        if grade_input == "":
            grade = 7.0
            break
        try:
            grade = float(grade_input)
            if 1 <= grade <= 13:
                break
            print("⚠️ Grade must be between 1 (poor) and 13 (mansion quality).")
        except ValueError:
            print("⚠️ Please enter a valid grade.")
            
    # Automatically lookup geographic coordinates and nearby sizes based on entered zipcode
    zip_data = df[df['zipcode'] == zipcode]
    lat = zip_data['lat'].mean()
    long = zip_data['long'].mean()
    sqft_living15 = zip_data['sqft_living15'].mean()
    sqft_lot15 = zip_data['sqft_lot15'].mean()
    
    # Infer standard attributes for remaining features
    floors = 1.0 if sqft_living < 1500 else (2.0 if sqft_living > 2500 else 1.5)
    waterfront = 0.0
    view = 0.0
    condition = 3.0
    sqft_above = sqft_living * 0.9
    sqft_basement = sqft_living * 0.1
    yr_renovated = 0
    sale_year = 2015
    sale_month = 6
    
    # Calculate engineered features
    house_age = max(0, sale_year - yr_built)
    is_renovated = 0
    years_since_renovation = house_age
    sqft_per_room = sqft_living / (bedrooms + bathrooms)
    has_basement = 1 if sqft_basement > 0 else 0
    sqft_above_ratio = sqft_above / (sqft_living + 1e-5)
    
    # Construct feature matrix dictionary mapping
    new_house = {feat: 0.0 for feat in feature_names}
    
    new_house['bedrooms'] = bedrooms
    new_house['bathrooms'] = bathrooms
    new_house['sqft_living'] = sqft_living
    new_house['sqft_lot'] = sqft_lot
    new_house['floors'] = floors
    new_house['waterfront'] = waterfront
    new_house['view'] = view
    new_house['condition'] = condition
    new_house['grade'] = grade
    new_house['sqft_above'] = sqft_above
    new_house['sqft_basement'] = sqft_basement
    new_house['yr_built'] = float(yr_built)
    new_house['yr_renovated'] = float(yr_renovated)
    new_house['lat'] = lat
    new_house['long'] = long
    new_house['sqft_living15'] = sqft_living15
    new_house['sqft_lot15'] = sqft_lot15
    new_house['sale_year'] = float(sale_year)
    new_house['sale_month'] = float(sale_month)
    new_house['house_age'] = float(house_age)
    new_house['is_renovated'] = float(is_renovated)
    new_house['years_since_renovation'] = float(years_since_renovation)
    new_house['sqft_per_room'] = float(sqft_per_room)
    new_house['has_basement'] = float(has_basement)
    new_house['sqft_above_ratio'] = float(sqft_above_ratio)
    
    zipcode_dummy = f"zipcode_{zipcode}"
    if zipcode_dummy in new_house:
        new_house[zipcode_dummy] = 1.0
        
    return new_house, {
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 
        'sqft_living': sqft_living, 'sqft_lot': sqft_lot,
        'grade': grade, 'yr_built': yr_built, 'zipcode': zipcode,
        'lat': lat, 'long': long
    }

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "data", "kc_house_data.csv")
    
    if not os.path.exists(save_path):
        print(f"Error: Dataset not found at {save_path}. Please run 'python3 main.py' first to download the data.")
        return

    # 1. Load the dataset
    df = pd.read_csv(save_path)
    
    # 2. Fit the preprocessing scaler on dataset
    print("Training standard regression model & loading preprocessing scales. Please wait...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # 3. Train the model
    model = train_sklearn_linear_regression(X_train, y_train)
    
    # 4. Get interactive inputs
    new_house, summary_dict = get_user_input(feature_names, df)
    
    # 5. Convert to DataFrame
    new_house_df = pd.DataFrame([new_house])
    
    # 6. Scale numerical features using the fitted scaler
    columns_to_scale = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
        'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
        'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
        'sqft_living15', 'sqft_lot15', 'sale_year', 'sale_month', 
        'house_age', 'years_since_renovation', 'sqft_per_room', 'sqft_above_ratio'
    ]
    cols_scaled = [col for col in columns_to_scale if col in new_house_df.columns]
    
    new_house_scaled = new_house_df.copy()
    new_house_scaled[cols_scaled] = scaler.transform(new_house_scaled[cols_scaled].astype(float))
    
    # 7. Model inference (Predict price)
    prediction_input = new_house_scaled.values.astype(float)
    predicted_price = model.predict(prediction_input)[0]
    
    # Negative predictions safeguard (extreme outliers)
    predicted_price = max(10000.0, predicted_price)
    
    # 8. Display Prediction Output
    print("\n" + "="*60)
    print("🏡 PREDICTION RESULTS FOR YOUR CUSTOM INPUTS")
    print("="*60)
    print(f"  * Bedrooms: {summary_dict['bedrooms']:.0f}  |  Bathrooms: {summary_dict['bathrooms']:.2f}")
    print(f"  * Living Area: {summary_dict['sqft_living']:,.0f} sqft  |  Lot Area: {summary_dict['sqft_lot']:,.0f} sqft")
    print(f"  * Quality Grade: {summary_dict['grade']:.0f}/13")
    print(f"  * Year Built: {summary_dict['yr_built']}")
    print(f"  * Zipcode Location: {summary_dict['zipcode']} (lat: {summary_dict['lat']:.4f}, long: {summary_dict['long']:.4f})")
    print("-"*60)
    print(f"  🔥 ESTIMATED HOUSE VALUE: ${predicted_price:,.2f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
