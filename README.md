# Fitbit Health Analytics Dashboard

End-to-end Fitbit analytics project using Python, SQL/SQLite, Streamlit and Plotly.

## Run
```bash
cd Fitbit_Health_Analytics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard pages
- Overview
- Activity Explorer
- Intensity & Calories
- Sleep Analytics
- Heart Rate
- Weight & BMI
- SQL Insights
- Data Explorer

## Interactive sidebar
Date range, participant selection, weekday filter, activity-level filter and rolling-average toggle.

## SQL
The SQLite database contains cleaned analytics tables plus reusable SQL views. See `sql/insights.sql`.

## Cleaning
Duplicates removed, dates standardized, numeric fields cleaned, invalid heart-rate readings outside 20–250 BPM removed, sleep duplicate keys removed, and minute-level heart-rate data aggregated to hourly/daily tables for dashboard performance.
