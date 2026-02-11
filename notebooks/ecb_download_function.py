import requests
import pandas as pd
import os

def download_ecb_to_csv():
    url = "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV"
    params = {'format': 'csvdata'}
    
    response = requests.get(url, params=params)
    
    # Define the file path
    file_path = "../data/raw/ecb_market_rates.csv"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save as Flat File
    with open(file_path, 'wb') as f:
        f.write(response.content)
    
    print(f"Flat file saved to {file_path}")
    return file_path

