# train_prophet_models.py
import os
import pandas as pd
from prophet import Prophet
import joblib

# ============================
# Load the Cleaned Dataset
# ============================
df = pd.read_csv("clean_india_air_quality.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date', 'City'])

# Get pollutant columns
pollutants = [col for col in df.columns if col not in ['City', 'Date']]

# Create directory to save models
os.makedirs("prophet_models", exist_ok=True)

# ============================
# Train Prophet Model per City + Pollutant
# ============================
for city in df['City'].unique():
    city_df = df[df['City'] == city]
    print(f"\n🏙️ Training models for city: {city}")

    for pollutant in pollutants:
        pollutant_df = city_df[['Date', pollutant]].dropna()
        if len(pollutant_df) < 30:
            print(f"   ⚠️ Skipping {pollutant} (insufficient data)")
            continue

        # Prepare data for Prophet
        data = pollutant_df.rename(columns={'Date': 'ds', pollutant: 'y'})

        # Train model
        model = Prophet(daily_seasonality=True)
        model.fit(data)

        # Save model
        model_path = f"prophet_models/{city}_{pollutant}_model.pkl"
        joblib.dump(model, model_path)
        print(f"   ✅ Model saved: {model_path}")

print("\n🎉 All available models have been trained and saved in the 'prophet_models/' folder.")
