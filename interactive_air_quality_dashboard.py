import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st

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
# 6. Prophet Forecast (dynamic)
# ----------------------------
st.subheader(f"🔮 Forecasting {selected_pollutant} for {city}")

def generate_forecast(city_df, selected_pollutant, forecast_days):
    data = city_df[['Date', selected_pollutant]].dropna().rename(columns={'Date':'ds', selected_pollutant:'y'})
    if len(data) <= 30:
        return None, None
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return forecast, model

forecast, model = generate_forecast(city_df, selected_pollutant, forecast_days)

if forecast is not None:
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    st.download_button("📥 Download Forecast Data", 
                       data=forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(index=False),
                       file_name=f"{city}_{selected_pollutant}_forecast.csv",
                       mime="text/csv")
else:
    st.warning("Not enough data for forecasting.")

# ----------------------------
# 7. Download Clean Dataset
# ----------------------------
st.subheader("📥 Download Dataset")
st.download_button("Download Clean India Air Quality Data", 
                   data=df.to_csv(index=False),
                   file_name="clean_india_air_quality.csv",
                   mime="text/csv")
