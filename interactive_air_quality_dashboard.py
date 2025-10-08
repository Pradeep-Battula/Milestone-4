import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st
import os
import joblib

# ----------------------------
# Load cleaned dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("clean_india_air_quality.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

df = load_data()

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Air Quality Dashboard - India")
city = st.sidebar.selectbox("Select City", sorted(df['City'].unique()))
pollutants = [col for col in df.columns if col not in ['City','Date']]
selected_pollutant = st.sidebar.selectbox("Select Pollutant (for Forecasting)", pollutants)
multi_pollutants = st.sidebar.multiselect("Compare Multiple Pollutants", pollutants, default=['PM2.5','PM10'])
forecast_days = st.sidebar.slider("Forecast Days Ahead", 7, 90, 30)

# ----------------------------
# Title
# ----------------------------
st.title("🌍 AirAware Smart Air Quality Prediction System")
st.write(f"### City: {city}")

city_df = df[df['City'] == city]

# Ensure pollutant columns are numeric
for pol in pollutants:
    city_df[pol] = pd.to_numeric(city_df[pol], errors='coerce')

latest = city_df.sort_values("Date").iloc[-1]
latest_pollutants = latest[pollutants].dropna()

# ----------------------------
# Compute overall AQI for donut
# ----------------------------
if not latest_pollutants.empty:
    def simple_aqi(val, pol):
        thresholds = {'PM2.5': 60, 'PM10': 100, 'NO2': 80, 'SO2': 80, 'O3': 100,
                      'CO': 10, 'NH3': 400, 'NO': 80, 'NOx': 100}
        return (val / thresholds.get(pol, 100)) * 100

    aqi_values = [simple_aqi(v, pol) for pol, v in latest_pollutants.items()]
    overall_aqi = int(max(aqi_values))
else:
    overall_aqi = None

# ----------------------------
# Function to create card style container
# ----------------------------
def card(content):
    st.markdown(f"""
        <div style="
            background-color:#1E1E1E;
            padding:20px;
            border-radius:15px;
            margin-bottom:15px;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
            width:100%;
        ">
            {content}
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# 1. Top Metrics Cards
# ----------------------------
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Donut chart
with col1:
    content = "<h4>🥯 Current Air Quality</h4>"
    if not latest_pollutants.empty:
        num_pollutants = len(latest_pollutants)
        wedge_width = max(0.2, 0.6 - num_pollutants*0.02)
        fig, ax = plt.subplots(figsize=(4,4), facecolor='none')
        wedges, texts, autotexts = ax.pie(
            latest_pollutants, labels=latest_pollutants.index,
            autopct="%1.1f%%", startangle=140,
            colors=plt.cm.tab20.colors[:len(latest_pollutants)],
            wedgeprops={'width':wedge_width, 'edgecolor':'w'},
            pctdistance=0.8
        )
        for text in texts + autotexts:
            text.set_color('white')
            text.set_fontsize(10)
            text.set_fontweight('bold')
        ax.text(0, 0, f"AQI\n{overall_aqi}", ha='center', va='center',
                fontsize=16, fontweight='bold', color='red')
        st.pyplot(fig, clear_figure=True)
    else:
        st.warning("No data for donut chart")

# Latest PM2.5 card
with col2:
    content = f"<h4>🌫️ Latest PM2.5</h4><h3>{latest['PM2.5']} µg/m³</h3>"
    card(content)

# Latest PM10 card
with col3:
    content = f"<h4>💨 Latest PM10</h4><h3>{latest['PM10']} µg/m³</h3>"
    card(content)

# Overall AQI card
with col4:
    content = f"<h4>📊 Overall AQI</h4><h3>{overall_aqi}</h3>"
    card(content)

# ----------------------------
# Dynamic Cards for Selected Pollutants
# ----------------------------
st.subheader("📌 Selected Pollutants")
if multi_pollutants:
    num_cols = min(3, len(multi_pollutants))  # max 3 columns
    cols = st.columns(num_cols)
    for i, pol in enumerate(multi_pollutants):
        val = latest.get(pol, "N/A")
        with cols[i % num_cols]:
            content = f"<h4>{pol}</h4><h3>{val}</h3>"
            card(content)
else:
    st.info("Select at least one pollutant to display dynamic cards.")

# ----------------------------
# 2. AQI Category Snapshot
# ----------------------------
def aqi_category(pollutant, value):
    if pd.isna(value): return "No Data"
    if pollutant == "PM2.5":
        if value <= 30: return "Good ✅"
        elif value <= 60: return "Satisfactory 🙂"
        elif value <= 90: return "Moderate 😐"
        elif value <= 120: return "Poor 😷"
        elif value <= 250: return "Very Poor 🤒"
        else: return "Severe ☠️"
    elif pollutant == "PM10":
        if value <= 50: return "Good ✅"
        elif value <= 100: return "Satisfactory 🙂"
        elif value <= 250: return "Moderate 😐"
        elif value <= 350: return "Poor 😷"
        elif value <= 430: return "Very Poor 🤒"
        else: return "Severe ☠️"
    elif pollutant == "NO2":
        if value <= 40: return "Good ✅"
        elif value <= 80: return "Satisfactory 🙂"
        elif value <= 180: return "Moderate 😐"
        elif value <= 280: return "Poor 😷"
        elif value <= 400: return "Very Poor 🤒"
        else: return "Severe ☠️"
    elif pollutant == "SO2":
        if value <= 40: return "Good ✅"
        elif value <= 80: return "Satisfactory 🙂"
        elif value <= 380: return "Moderate 😐"
        elif value <= 800: return "Poor 😷"
        elif value <= 1600: return "Very Poor 🤒"
        else: return "Severe ☠️"
    elif pollutant == "O3":
        if value <= 50: return "Good ✅"
        elif value <= 100: return "Satisfactory 🙂"
        elif value <= 168: return "Moderate 😐"
        elif value <= 208: return "Poor 😷"
        elif value <= 748: return "Very Poor 🤒"
        else: return "Severe ☠️"
    elif pollutant == "CO":
        if value <= 1: return "Good ✅"
        elif value <= 2: return "Satisfactory 🙂"
        elif value <= 10: return "Moderate 😐"
        elif value <= 17: return "Poor 😷"
        elif value <= 34: return "Very Poor 🤒"
        else: return "Severe ☠️"
    elif pollutant == "NH3":
        if value <= 200: return "Good ✅"
        elif value <= 400: return "Satisfactory 🙂"
        elif value <= 800: return "Moderate 😐"
        elif value <= 1200: return "Poor 😷"
        elif value <= 1800: return "Very Poor 🤒"
        else: return "Severe ☠️"
    return "No Category"

st.subheader("🏷️ AQI Category by Pollutant")
aqi_results = {poll: aqi_category(poll, latest.get(poll, None)) for poll in pollutants}
st.table(pd.DataFrame.from_dict(aqi_results, orient="index", columns=["AQI Category"]))

# ----------------------------
# 3. Daily Average Trends
# ----------------------------
st.subheader("📊 Daily Average Trends")
daily_avg = city_df.groupby('Date')[pollutants].mean().reset_index()

fig, ax = plt.subplots(figsize=(12,6))
plotted = False
for pol in multi_pollutants:
    if pol in daily_avg.columns and daily_avg[pol].notna().sum() > 0:
        ax.plot(daily_avg['Date'], daily_avg[pol], label=pol)
        plotted = True

if plotted:
    ax.set_title(f"Daily Average Pollutants in {city}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Concentration")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("⚠️ Select any two pollutants to show daily average trends.")

# ----------------------------
# 4. Normalized Multi-Pollutant Comparison
# ----------------------------
st.subheader("📈 Normalized Multi-Pollutant Comparison")
norm_df = daily_avg.copy()
for pol in multi_pollutants:
    norm_df[pol] = (norm_df[pol] - norm_df[pol].min()) / (norm_df[pol].max() - norm_df[pol].min())

fig, ax = plt.subplots(figsize=(12,6))
for pol in multi_pollutants:
    ax.plot(norm_df['Date'], norm_df[pol], label=pol)
ax.set_title(f"Normalized Daily Trends in {city}")
ax.set_xlabel("Date")
ax.set_ylabel("Normalized Value (0-1)")
ax.legend()
st.pyplot(fig)

# ----------------------------
# 5. Alerts for High Pollution
# ----------------------------
st.subheader("⚠️ Alerts for Exceeding Safe Limits")
thresholds = {'PM2.5': 60, 'PM10': 100, 'NO2': 80, 'SO2': 80, 'O3': 100}
for pol in pollutants:
    latest_val = latest.get(pol)
    if latest_val and pol in thresholds and latest_val > thresholds[pol]:
        st.warning(f"{pol} exceeds safe limit! ({latest_val})")

# ----------------------------
# 6. Multi-Pollutant Prophet Forecast & Predicted AQI
# ----------------------------
st.subheader(f"🔮 Forecasting Selected Pollutants for {city}")

# Filter only pollutants that have trained models
available_pollutants = [p for p in multi_pollutants if os.path.exists(f"prophet_models/{city}_{p}_model.pkl")]
skipped_pollutants = [p for p in multi_pollutants if p not in available_pollutants]

if skipped_pollutants:
    st.warning(f"⚠️ No trained models found for: {', '.join(skipped_pollutants)}. Please retrain them.")

if available_pollutants:
    forecast_results = {}  # Store forecasts for all available pollutants
    for pol in available_pollutants:
        model_path = f"prophet_models/{city}_{pol}_model.pkl"
        model = joblib.load(model_path)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        forecast_results[pol] = forecast

        # Plot forecast
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='orange')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2)
        ax.set_title(f"{pol} Forecast for {city}")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{pol} Concentration")
        ax.legend()
        st.pyplot(fig)

    # Predicted AQI for multiple pollutants
    st.subheader(f"📈 Predicted AQI Trend based on Selected Pollutants")
    predicted_aqi_df = pd.DataFrame({'Date': forecast_results[available_pollutants[0]]['ds']})

    thresholds = {'PM2.5': 60, 'PM10': 100, 'NO2': 80, 'SO2': 80, 'O3': 100,
                  'CO': 10, 'NH3': 400, 'NO': 80, 'NOx': 100}

    for pol in available_pollutants:
        forecasted_aqi = forecast_results[pol]['yhat'] / thresholds.get(pol, 100) * 100
        predicted_aqi_df[f"{pol} AQI (%)"] = forecasted_aqi

    # Overall predicted AQI as max of all pollutants
    predicted_aqi_df['Overall Predicted AQI (%)'] = predicted_aqi_df[[f"{p} AQI (%)" for p in available_pollutants]].max(axis=1)

    # Plot overall predicted AQI
    fig_aqi, ax_aqi = plt.subplots(figsize=(10,5))
    ax_aqi.plot(predicted_aqi_df['Date'], predicted_aqi_df['Overall Predicted AQI (%)'], color='purple', label='Overall Predicted AQI')
    ax_aqi.axhline(100, color='red', linestyle='--', label='Safe Limit (100%)')
    ax_aqi.set_xlabel("Date")
    ax_aqi.set_ylabel("Predicted AQI (%)")
    ax_aqi.set_title(f"Overall Predicted AQI Trend for {city}")
    ax_aqi.legend()
    st.pyplot(fig_aqi)

    # Show table for forecast days ahead
    st.subheader(f"Next {forecast_days} Days Predicted AQI")
    st.table(predicted_aqi_df.head(forecast_days))
else:
    st.info("No trained models available for the selected pollutants. Please upload a dataset and retrain models first.")

# ----------------------------
# 7. Admin Panel in Sidebar
# ----------------------------
with st.sidebar.expander("🛠️Upload New Dataset & Retrain Models", expanded=False):
    st.info("Upload a cleaned dataset and retrain Prophet models for all cities & pollutants.")
    uploaded_file = st.file_uploader("📤 Upload Cleaned Air Quality Dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        new_df['Date'] = pd.to_datetime(new_df['Date'], errors='coerce')
        st.success(f"✅ Uploaded dataset with {len(new_df)} records and {len(new_df.columns)} columns.")
        if st.button("🚀 Retrain Prophet Models"):
            with st.spinner("Training models... please wait ⏳"):
                pollutants_new = [col for col in new_df.columns if col not in ['City', 'Date']]
                os.makedirs("prophet_models", exist_ok=True)
                total_models = 0
                for city_name in new_df['City'].dropna().unique():
                    city_df_new = new_df[new_df['City'] == city_name]
                    st.write(f"🏙️ Training models for **{city_name}**...")
                    for pollutant in pollutants_new:
                        data = city_df_new[['Date', pollutant]].dropna()
                        if len(data) < 30:
                            st.warning(f"⚠️ Skipping {pollutant} for {city_name} (insufficient data)")
                            continue
                        df_train = data.rename(columns={'Date': 'ds', pollutant: 'y'})
                        model = Prophet(daily_seasonality=True)
                        model.fit(df_train)
                        model_path = f"prophet_models/{city_name}_{pollutant}_model.pkl"
                        joblib.dump(model, model_path)
                        total_models += 1
                        st.success(f"✅ Saved: {model_path}")
                st.success(f"🎉 Retraining completed. {total_models} models saved successfully.")

# ----------------------------
# 8. Download Clean Dataset
# ----------------------------
st.subheader("📥 Download Dataset")
st.download_button("Download Clean India Air Quality Data", 
                   data=df.to_csv(index=False),
                   file_name="clean_india_air_quality.csv",
                   mime="text/csv")
