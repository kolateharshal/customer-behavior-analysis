import os
import urllib.request
import pandas as pd

def download_and_load_data(url: str, save_path: str) -> pd.DataFrame:
    """
    Downloads the dataset from the given URL and saves it locally.
    Loads it into a pandas DataFrame and returns it.
    
    Parameters:
    -----------
    url : str
        The URL to download the dataset CSV from.
    save_path : str
        The local file path where the CSV should be saved.
        
    Returns:
    --------
    pd.DataFrame
        The loaded pandas DataFrame containing the house sales data.
    """
    # Create the directory if it doesn't exist
    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")
    
    # Download the file if it doesn't exist
    if not os.path.exists(save_path):
        print(f"Downloading dataset from {url}...")
        try:
            # Using urllib to retrieve the dataset
            urllib.request.urlretrieve(url, save_path)
            print(f"Dataset successfully downloaded and saved to {save_path}")
        except Exception as e:
            print(f"Failed to download the dataset. Error: {e}")
            raise e
    else:
        print(f"Dataset already exists at {save_path}. Skipping download.")
        
    # Load and return the dataset
    df = pd.read_csv(save_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Test script execution
    DATA_URL = "https://raw.githubusercontent.com/dhyan6/data-science-projects/main/kc_house_data.csv"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(os.path.dirname(current_dir), "data", "kc_house_data.csv")
    download_and_load_data(DATA_URL, SAVE_PATH)
