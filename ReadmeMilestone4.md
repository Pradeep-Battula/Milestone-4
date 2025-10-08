# AirAware Smart Air Quality Dashboard – Milestone 4

## **Module:** Streamlit Web Dashboard

**Duration:** Weeks 7–8

---

## **Overview**

Milestone 4 focuses on developing an **interactive web dashboard** using Streamlit to monitor and forecast air quality. The dashboard allows users to:

* Select a city/station for monitoring
* Choose pollutants for trend analysis
* Forecast pollutant levels for future days
* View AQI in a dynamic, visual format
* Receive alerts for pollutants exceeding safe limits

The dashboard provides an intuitive interface for **both visualization and actionable insights**.

---

## **Key Features**

### **1. Interactive UI**

* Users can select:

  * **City / Station**
  * **Pollutant for Forecasting**
  * **Multiple Pollutants for Comparison**
  * **Forecast Days Ahead**
* Sidebar controls provide a simple way to interact with data dynamically.

### **2. AQI Gauge / Donut Chart**

* Displays **current AQI and contribution of pollutants** in a donut chart.
* Overall AQI value is highlighted at the center.
* Text and percentage labels are dynamically readable.

### **3. Line Plots for Historical Trends**

* **Daily average trends** of selected pollutants.
* **Normalized trend comparison** for multiple pollutants.
* Interactive and updates based on user selection.

### **4. Alerts for High Pollution**

* Highlights pollutants exceeding safe limits.
* Dynamic warning messages based on latest data.

### **5. Forecasting**

* Users can forecast selected pollutant levels for a user-defined number of days.
* Uses **Prophet model** for time-series forecasting.
* Provides:

  * Forecast plot
  * Component plots (trend, seasonality)
  * CSV download for forecast data

### **6. Data Management**

* Download the clean dataset for offline use or admin purposes.
* (Optional) Extendable for uploading new data and retraining models.

---

## **Installation & Usage**

### **Prerequisites**

* Python 3.9+
* Install required libraries:


pip install pandas matplotlib prophet streamlit


### **Running the Dashboard**

1. Place the cleaned dataset (`clean_india_air_quality.csv`) in the project folder.
2. Run Streamlit:


streamlit run app.py


3. Access the dashboard in your browser at `http://localhost:8501`.

---

## **Dashboard Components**

| Component              | Description                                                                     |
| ---------------------- | ------------------------------------------------------------------------------- |
| Sidebar Controls       | City selection, pollutant selection, forecast days, multi-pollutant selection   |
| AQI Donut Chart        | Visual representation of current AQI and pollutant contributions                |
| Latest Pollutant Cards | Displays latest values of selected pollutants                                   |
| Alerts                 | Highlights pollutants exceeding safe limits                                     |
| Trend Plots            | Shows daily average and normalized trends of selected pollutants                |
| Forecast               | Predicts future pollutant levels using Prophet, with plots and downloadable CSV |
| Dataset Download       | Provides access to the cleaned dataset for analysis                             |

---

## **Code Highlights**

### **Interactive UI**


city = st.sidebar.selectbox("Select City", sorted(df['City'].unique()))
selected_pollutant = st.sidebar.selectbox("Select Pollutant (for Forecasting)", pollutants)
multi_pollutants = st.sidebar.multiselect("Compare Multiple Pollutants", pollutants, default=['PM2.5','PM10'])
forecast_days = st.sidebar.slider("Forecast Days Ahead", 7, 90, 30)


### **AQI Donut Chart**


fig, ax = plt.subplots(figsize=(4,4), facecolor='none')
wedges, texts, autotexts = ax.pie(
    latest_pollutants, labels=latest_pollutants.index,
    autopct="%1.1f%%", startangle=140,
    colors=plt.cm.tab20.colors[:len(latest_pollutants)],
    wedgeprops={'width':0.4, 'edgecolor':'w'},
    pctdistance=0.8
)
ax.text(0, 0, f"AQI\n{overall_aqi}", ha='center', va='center',
        fontsize=16, fontweight='bold', color='red')
st.pyplot(fig, clear_figure=True)


### **Alerts for High Pollution**


thresholds = {'PM2.5': 60, 'PM10': 100, 'NO2': 80, 'SO2': 80, 'O3': 100}
for pol in pollutants:
    latest_val = latest.get(pol)
    if latest_val and pol in thresholds and latest_val > thresholds[pol]:
        st.warning(f"{pol} exceeds safe limit! ({latest_val})")


### **Forecasting**


model = Prophet(daily_seasonality=True)
model.fit(forecast_data)
future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)
fig1 = model.plot(forecast)
st.pyplot(fig1)


---

## **Future Improvements**

* Admin interface to **upload new datasets** and retrain models directly from the dashboard.
* Integration of **real-time AQI data APIs**.
* Enhanced UI with **cards for selected pollutants** dynamically.

---

✅ **Milestone 4 Completion Status:**
All core requirements of the Streamlit Web Dashboard have been successfully implemented.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

