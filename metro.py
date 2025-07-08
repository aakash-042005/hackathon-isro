import requests
import pandas as pd

locations = {
    "Arumbakkam": (13.0736, 80.2022),
    "Bhopal": (23.2599, 77.4126),
    "Mumbai": (19.0760, 72.8777),
    "West_Bengal": (22.5726, 88.3639),
    "Anand_Vihar": (28.6506, 77.3150)
}

base_url = "https://archive-api.open-meteo.com/v1/archive"
start_date = "2024-01-01"
end_date = "2024-06-30"
daily_vars = "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,relative_humidity_2m_max"

dfs = []

for name, (lat, lon) in locations.items():
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": daily_vars,
        "timezone": "Asia/Kolkata"
    }
    r = requests.get(base_url, params=params)
    data = r.json()["daily"]
    df = pd.DataFrame(data)
    df["location"] = name
    dfs.append(df)

combined = pd.concat(dfs)
combined.to_csv("combined_meteorological_data.csv", index=False)
print("Saved: combined_meteorological_data.csv")
