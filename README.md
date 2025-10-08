# 🌍 AirAware Smart Air Quality Prediction System

AirAware is a **smart, interactive dashboard** that provides **real-time and forecasted air quality insights** for cities in India. It leverages historical air quality data and **Prophet time series forecasting models** to predict future pollutant concentrations and the overall AQI. This system also includes an **admin panel** to upload new datasets and retrain models for improved predictions.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Admin Panel](#admin-panel)
- [Technologies Used](#technologies-used)
- [Data Flow Diagram](#data-flow-diagram)
- [License](#license)
- [Author](#author)

---

## Features

### 1. City-Level Air Quality Dashboard
- View **current pollutant levels** and **overall AQI** in a visually appealing **donut chart**.
- Compare multiple pollutants with dynamic cards.
- Check **daily average trends** and **normalized multi-pollutant comparisons**.

### 2. AQI Category Snapshot
- Provides **AQI categories** (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) for each pollutant based on the latest readings.

### 3. Forecasting & Predicted AQI
- **Prophet time series models** predict future pollutant levels.
- Supports **multi-pollutant forecasting** for selected pollutants.
- Generates **predicted AQI trends** for the chosen forecast period.
- Highlights **high-risk days** exceeding safe limits.

### 4. Alerts
- ⚠️ Automatic alerts when pollutant levels exceed predefined safe thresholds.

### 5. Admin Panel
- Upload new cleaned air quality datasets (CSV).
- Retrain **Prophet models** for all cities and pollutants.
- Models are automatically saved in `prophet_models/`.

### 6. Data Download
- Download **clean dataset** and **forecasted pollutant data** in CSV format.

---

## Project Structure

AirAware/
│
├── clean_india_air_quality.csv # Cleaned dataset
├── interactive_air_quality_dashboard.py # Main Streamlit app
├── train_prophet_models.py # Optional training script
├── prophet_models/ # Folder for trained Prophet models
├── LICENSE # MIT License
└── README.md # Project documentation

yaml
Copy code

---

## Installation

1. Clone the repository:

git clone https://github.com/Pradeep-Battula/Milestone-4.git
cd Milestone-4

Install required Python packages:
pip install pandas matplotlib prophet streamlit joblib

Run the dashboard:
streamlit run interactive_air_quality_dashboard.py


## Usage
Select City: Choose the city to view air quality data.

Select Pollutants: Pick one pollutant for forecast or multiple for comparison.

Forecast Days Ahead: Set the number of future days for predictions.

View Dashboard: Monitor current AQI, daily trends, normalized pollutant comparison, and predicted AQI.

Download Data: Save clean data or forecasted results as CSV files.

Admin Panel
Located in the sidebar below the forecast section.

Upload a new CSV file containing cleaned air quality data.

Click Retrain Prophet Models to update all city-level models.

Models are automatically stored in prophet_models/.

## Technologies Used

Python 3.11

Streamlit for interactive web app

Prophet for time series forecasting

Pandas & Matplotlib for data processing and visualization

Joblib for model serialization

## Data Flow Diagram

<img width="721" height="431" alt="image" src="https://github.com/user-attachments/assets/e13dacee-5bcc-4a17-a36d-b98f46474d1a" />


## License
This project is licensed under the MIT License. See LICENSE(LICENSE) for details.
