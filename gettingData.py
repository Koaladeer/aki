import os

import pandas as pd
import requests
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime


# API-Schlüssel und Basis-URL
API_KEY = "PKS2735LEPXOLUV8MFDM"
SECRET_KEY = "EPLKx7mQseQfJPkYxLoK6iQeqbvpLtK3XKI2nRW9"
BASE_URL = "https://paper-api.alpaca.markets/"

# Verbindung einrichten
api = REST(API_KEY, SECRET_KEY, BASE_URL)

def fetch_historical_data(symbol, start_date, end_date, save_path):

    savePathDiracetory = (save_path.rsplit('/', 1)[0])
    if not os.path.exists(savePathDiracetory):
        os.makedirs(savePathDiracetory)
    historical_data = api.get_bars(
        symbol, TimeFrame.Day, start=start_date, end=end_date
    ).df
    historical_data.to_csv(save_path)
    print(f"Historische Daten für {symbol} gespeichert: {save_path}")

def fetch_latest_price(symbol):
    latest_trade = api.get_latest_trade(symbol)
    today_date_time = datetime.now().strftime("%Y-%m-%d %H:%M")  # Format: YYYY-MM-DD HH:MM
    # print("Today's date:", today_date)
    print(f"Aktuelle Daten für {symbol}: {latest_trade}")
def fetch_bayer_trials_to_csv(output_csv="bayer_trials.csv"): ##todo: not working! change that
    """
    Fetches clinical trials data sponsored by Bayer from ClinicalTrials.gov,
    extracts specific fields, and writes them to a CSV file.

    Args:
        output_csv (str): The file path for the output CSV. Default is 'bayer_trials.csv'.
    """
    #https://clinicaltrials.gov/data-api/api to fix
    # Base URL for the ClinicalTrials.gov API
    base_url = "https://clinicaltrials.gov/api/query/full_studies"
    query_params = {
        "spons": "Bayer",  # Sponsor name
        "aggFilters": "phase:1 2 3,status:com",  # Phases 1, 2, 3 and completed status
    }
    request_url = f"{base_url}?" + "&".join(f"{key}={value}" for key, value in query_params.items())
    print("Request URL:", request_url)
    # Fetch data from ClinicalTrials.gov
    response = requests.get(base_url, params=query_params)
    if response.status_code != 200:
        print("Failed to fetch data. HTTP Status:", response.status_code)
        print("")
        return

    # Parse JSON response
    data = response.json()
    if not data.get("FullStudiesResponse") or not data["FullStudiesResponse"].get("FullStudies"):
        print("No studies found.")
        return

    studies = data["FullStudiesResponse"]["FullStudies"]

    # Extract required fields from the studies
    trials = []
    for study in studies:
        try:
            protocol = study["Study"]["ProtocolSection"]
            trial = {
                "Study Start (Actual)": protocol.get("StartDateStruct", {}).get("StartDate"),
                "Primary Completion (Actual)": protocol.get("PrimaryCompletionDateStruct", {}).get("PrimaryCompletionDate"),
                "Study Completion (Actual)": protocol.get("CompletionDateStruct", {}).get("CompletionDate"),
                "Phase": protocol.get("DesignModule", {}).get("PhaseList", {}).get("Phase"),
                "Sponsor": protocol.get("SponsorCollaboratorsModule", {}).get("LeadSponsor", {}).get("LeadSponsorName")
            }
            trials.append(trial)
        except KeyError as e:
            print(f"Missing data for a study: {e}")
            continue

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(trials)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")
if __name__ == "__main__":
    fetch_historical_data("AAPL", "2022-01-01", "2022-12-31", "Data/raw_data.csv")
    fetch_latest_price("AAPL")

    ##CAREFUL BUGGY
    fetch_bayer_trials_to_csv("bayer_trials.csv")

