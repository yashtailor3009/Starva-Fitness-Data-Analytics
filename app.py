from pathlib import Path
import base64
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# PROJECT PATHS / LOGO

ROOT = Path(__file__).resolve().parent
LOGO = ROOT / "starva_logo_bright.png"
LOGO_DATA = base64.b64encode(LOGO.read_bytes()).decode("utf-8") if LOGO.exists() else ""

# STARVA FITNESS DATA ANALYTICS
# Streamlit + SQLite + SQL + Plotly + Matplotlib + Seaborn

st.set_page_config(
    page_title="Starva Fitness Data Analytics",
    page_icon=str(LOGO) if LOGO.exists() else "🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# THEME

st.markdown(
    """
<style>
.stApp {
    background-color:#060913 !important;
    color:#e2e8f0;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}
[data-testid="stSidebar"] {
    background-color:#080d1a !important;
    border-right:1px solid #172554 !important;
}
[data-testid="stSidebar"] * { color:#e2e8f0; }

.hero {
    display:flex;
    justify-content:space-between;
    align-items:center;
    background:linear-gradient(135deg,#0d172b,#0b1220);
    padding:20px 25px;
    border-radius:16px;
    border:1px solid #0ea5e9;
    box-shadow:0 0 0 1px rgba(14,165,233,.08), 0 14px 35px rgba(0,0,0,.22);
    margin-bottom:20px;
}
.brand-title {
    color:#f8fafc;
    font-size:28px;
    font-weight:900;
    margin:0;
}
.brand-sub {
    color:#94a3b8;
    font-size:13px;
    margin-top:5px;
}
.status-pill {
    color:#86efac;
    font-size:13px;
    font-weight:700;
    background:#052e1b;
    border:1px solid #166534;
    border-radius:20px;
    padding:8px 13px;
}
.kpi-box {
    background:#0f172a;
    padding:15px 18px;
    border-radius:12px;
    border:1px solid #1e293b;
    min-height:105px;
}
.kpi-accent-green { border-top:3px solid #22c55e; }
.kpi-accent-cyan { border-top:3px solid #06b6d4; }
.kpi-accent-amber { border-top:3px solid #f59e0b; }
.kpi-accent-purple { border-top:3px solid #a855f7; }
.kpi-accent-blue { border-top:3px solid #3b82f6; }
.kpi-accent-rose { border-top:3px solid #f43f5e; }
.kpi-title {
    font-size:11px;
    font-weight:700;
    color:#94a3b8;
    text-transform:uppercase;
    letter-spacing:.5px;
}
.kpi-val {
    font-size:24px;
    font-weight:800;
    color:white;
    margin-top:4px;
}
.kpi-desc {
    font-size:11px;
    color:#22c55e;
    margin-top:2px;
}
.section-card {
    background:#0d1527;
    border:1px solid #1e293b;
    border-radius:14px;
    padding:16px;
}
.insight-card {
    background:linear-gradient(135deg,#064e3b 0%,#022c22 100%);
    border:1px solid #10b981;
    border-radius:14px;
    padding:18px;
    color:#e2e8f0;
}
.warning-card {
    background:#241a05;
    border:1px solid #a16207;
    border-radius:14px;
    padding:16px;
}
.small-muted {
    color:#94a3b8;
    font-size:12px;
}
/* ============================================================
   STARVA PROJECT LOGO
   ============================================================ */
.project-logo-card {
    display:flex;
    background:linear-gradient(135deg,rgba(14,165,233,.08),rgba(255,98,27,.05));
    border-radius:14px;
    align-items:center;
    gap:12px;
    padding:6px 2px 12px 2px;
}
.project-logo-img {
    width:78px;
    height:78px;
    object-fit:contain;
    flex-shrink:0;
}
.project-logo-title {
    color:#f8fafc;
    font-size:18px;
    font-weight:900;
    line-height:1.05;
}
.project-logo-sub {
    color:#94a3b8;
    font-size:9px;
    font-weight:700;
    letter-spacing:.65px;
    text-transform:uppercase;
    margin-top:5px;
}
.hero-brand {
    display:flex;
    align-items:center;
    gap:15px;
}
.hero-logo-img {
    width:105px;
    height:105px;
    object-fit:contain;
    flex-shrink:0;
    filter:drop-shadow(0 0 14px rgba(255,98,27,.18));
}

</style>
""",
    unsafe_allow_html=True,
)

# DATABASE

DB = ROOT / "fitbit_analytics.db"

@st.cache_resource
def get_connection():
    if not DB.exists():
        st.error(
            f"SQLite database not found: {DB}\n\n"
            "Place fitbit_analytics.db in the same folder as app.py."
        )
        st.stop()
    return sqlite3.connect(DB, check_same_thread=False)

@st.cache_data
def load_table(name):
    return pd.read_sql_query(f'SELECT * FROM "{name}"', get_connection())

daily = load_table("daily_activity_clean")
hourly = load_table("hourly_activity_clean")
sleep = load_table("sleep_clean")
hr = load_table("heart_rate_hourly_clean")
weight = load_table("weight_clean")
participants = load_table("participant_summary")

# DATA TYPES / DERIVED COLUMNS

for df, col in [
    (daily, "Date"),
    (hourly, "DateHour"),
    (sleep, "Date"),
    (hr, "DateHour"),
    (weight, "DateOnly"),
]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "Id" in daily.columns:
    daily["Id"] = daily["Id"].astype(str)

for df in [hourly, sleep, hr, weight, participants]:
    if "Id" in df.columns:
        df["Id"] = df["Id"].astype(str)

# Ensure activity categories exist.
if "ActivityLevel" not in daily.columns:
    daily["ActivityLevel"] = np.select(
        [
            daily["VeryActiveMinutes"].fillna(0) >= 30,
            (
                daily["VeryActiveMinutes"].fillna(0)
                + daily["FairlyActiveMinutes"].fillna(0)
            ) >= 30,
        ],
        ["High", "Moderate"],
        default="Low",
    )

# Total active minutes = light + fair + very active.
daily["TotalActiveMinutes"] = (
    daily["VeryActiveMinutes"].fillna(0)
    + daily["FairlyActiveMinutes"].fillna(0)
    + daily["LightlyActiveMinutes"].fillna(0)
)

# Stable weekday order.
WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]
WEEKDAY_ORDER = {day: i for i, day in enumerate(WEEKDAYS)}

# HELPERS

def fmt(value, decimals=0):
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"

def render_kpi(label, value, sub="", accent="green"):
    st.markdown(
        f"""
        <div class="kpi-box kpi-accent-{accent}">
            <div class="kpi-title">{label}</div>
            <div class="kpi-val">{value}</div>
            <div class="kpi-desc">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def style_fig(fig, height=350):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=11),
        title=dict(font=dict(size=15, color="#f8fafc")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=10),
        ),
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
    )
    return fig

def show_plot(fig, height=350):
    st.plotly_chart(
        style_fig(fig, height),
        use_container_width=True,
        config={"displaylogo": False},
    )

def filtered_by_ids(df, ids):
    if "Id" not in df.columns:
        return df.copy()
    return df[df["Id"].astype(str).isin([str(x) for x in ids])].copy()

def safe_mean(df, col):
    return df[col].mean() if col in df.columns and not df.empty else np.nan

def sql_df(query):
    return pd.read_sql_query(query, get_connection())

# SIDEBAR

st.sidebar.markdown(
    f"""
    <div class="project-logo-card">
        <img class="project-logo-img" src="data:image/png;base64,{LOGO_DATA}" />
        <div>
            <div class="project-logo-title">Starva</div>
            <div class="project-logo-sub">Fitness Data Analytics</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Historical Fitness & Wellness Data Analytics")

pages = [
    "🏠 Overview",
    "🏃 Activity Explorer",
    "👥 Participant Analysis",
    "🔥 Intensity & Calories",
    "😴 Sleep Analytics",
    "❤️ Heart Rate",
    "⚖️ Weight & BMI",
    "🧹 Data Quality",
    "🧠 SQL Insights",
    "📊 Python EDA",
    "💡 Insights & Recommendations",
    "🔎 Data Explorer",
]

page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.subheader("Interactive Filters")

mn = daily["Date"].min().date()
mx = daily["Date"].max().date()

dr = st.sidebar.date_input(
    "Date Range",
    (mn, mx),
    min_value=mn,
    max_value=mx,
)

if isinstance(dr, tuple) and len(dr) == 2:
    start = pd.Timestamp(dr[0])
    end = pd.Timestamp(dr[1])
else:
    start = pd.Timestamp(mn)
    end = pd.Timestamp(mx)

all_ids = sorted(daily["Id"].dropna().unique())
ids = st.sidebar.multiselect(
    "Participants",
    all_ids,
    default=all_ids,
)

if not ids:
    ids = all_ids

day_options = ["All"] + WEEKDAYS
day = st.sidebar.selectbox("Day of Week", day_options)

levels = st.sidebar.multiselect(
    "Activity Level",
    ["Low", "Moderate", "High"],
    default=["Low", "Moderate", "High"],
)

rolling = st.sidebar.toggle("7-day rolling trend", True)

# FILTER DATA

fd = daily[
    daily["Date"].between(start, end)
    & daily["Id"].astype(str).isin([str(x) for x in ids])
    & daily["ActivityLevel"].isin(levels)
].copy()

if day != "All" and "DayName" in fd.columns:
    fd = fd[fd["DayName"] == day]

fh = filtered_by_ids(hourly, ids)
if "DateHour" in fh.columns:
    fh = fh[
        fh["DateHour"].between(
            start,
            end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
        )
    ]

fs = filtered_by_ids(sleep, ids)
if "Date" in fs.columns:
    fs = fs[fs["Date"].between(start, end)]

fhr = filtered_by_ids(hr, ids)
if "DateHour" in fhr.columns:
    fhr = fhr[
        fhr["DateHour"].between(
            start,
            end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
        )
    ]

fw = filtered_by_ids(weight, ids)
if "DateOnly" in fw.columns:
    fw = fw[fw["DateOnly"].between(start, end)]

# HEADER

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-brand">
            <img class="hero-logo-img" src="data:image/png;base64,{LOGO_DATA}" />
            <div>
                <div class="brand-title">Starva Fitness Data Analytics</div>
                <div class="brand-sub">
                    Interactive historical analysis of activity, calories, sleep,
                    heart rate and weight
                </div>
            </div>
        </div>
        <div class="status-pill">● Analytics Ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# OVERVIEW

if page == "🏠 Overview":

    st.subheader("📊 Executive Overview")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        render_kpi(
            "Total Steps",
            fmt(fd["TotalSteps"].sum()),
            "Selected records",
            "green",
        )
    with c2:
        render_kpi(
            "Calories",
            fmt(fd["Calories"].sum()),
            "Total recorded",
            "cyan",
        )
    with c3:
        render_kpi(
            "Avg Steps / Day",
            fmt(fd["TotalSteps"].mean()),
            "Daily average",
            "amber",
        )
    with c4:
        render_kpi(
            "Participants",
            fmt(len(ids)),
            "Selected users",
            "purple",
        )
    with c5:
        render_kpi(
            "Avg Sleep",
            fmt(fs["SleepHours"].mean(), 1) + " hrs",
            "Logged sleep",
            "blue",
        )
    with c6:
        render_kpi(
            "Avg Heart Rate",
            fmt(fhr["AvgHeartRate"].mean(), 1) + " BPM",
            "Recorded average",
            "rose",
        )

    st.markdown("### 📈 Core Activity Trends")

    a, b = st.columns([1.5, 1])

    with a:
        tr = (
            fd.groupby("Date", as_index=False)
            .agg(Steps=("TotalSteps", "mean"))
            .sort_values("Date")
        )

        if rolling and len(tr) > 2:
            tr["Steps_7D"] = tr["Steps"].rolling(7, min_periods=1).mean()

        y_cols = ["Steps"]
        if "Steps_7D" in tr.columns:
            y_cols.append("Steps_7D")

        fig = px.line(
            tr,
            x="Date",
            y=y_cols,
            markers=True,
            title="Daily Average Steps",
            color_discrete_sequence=["#22c55e", "#06b6d4"],
        )
        show_plot(fig, 350)

    with b:
        d = (
            fd.groupby("DayName", as_index=False)
            .agg(TotalSteps=("TotalSteps", "mean"))
        )
        d["order"] = d["DayName"].map(WEEKDAY_ORDER)
        d = d.sort_values("order")

        fig = px.bar(
            d,
            x="DayName",
            y="TotalSteps",
            title="Average Steps by Weekday",
            color_discrete_sequence=["#22c55e"],
            text_auto=".0f",
        )
        show_plot(fig, 350)

    a, b = st.columns(2)

    with a:
        d = fd.groupby("ActivityLevel", as_index=False).size()
        fig = px.pie(
            d,
            names="ActivityLevel",
            values="size",
            hole=.58,
            title="Activity Level Distribution",
            color="ActivityLevel",
            color_discrete_map={
                "Low": "#f43f5e",
                "Moderate": "#f59e0b",
                "High": "#22c55e",
            },
        )
        show_plot(fig, 330)

    with b:
        d = (
            fh.groupby("Hour", as_index=False)
            .agg(Steps=("StepTotal", "mean"))
        )
        fig = px.bar(
            d,
            x="Hour",
            y="Steps",
            title="Average Hourly Steps",
            color_discrete_sequence=["#84cc16"],
        )
        show_plot(fig, 330)

    a, b = st.columns([1.3, 1])

    with a:
        corr_cols = [
            c for c in [
                "TotalSteps",
                "Calories",
                "TotalDistance",
                "TotalActiveMinutes",
                "SedentaryMinutes",
            ]
            if c in fd.columns
        ]

        if len(corr_cols) >= 2:
            fig = px.imshow(
                fd[corr_cols].corr(),
                text_auto=".2f",
                title="Activity Correlation Heatmap",
                color_continuous_scale="Greens",
                aspect="auto",
            )
            show_plot(fig, 380)

    with b:
        avg_active = fd["TotalActiveMinutes"].mean()
        avg_sedentary = fd["SedentaryMinutes"].mean()

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">📌 Selected-period summary</div>
                <p>
                <b>{fmt(fd["TotalSteps"].mean())}</b> average steps/day<br>
                <b>{fmt(avg_active, 1)}</b> average active minutes/day<br>
                <b>{fmt(avg_sedentary, 1)}</b> average sedentary minutes/day<br>
                <b>{fmt(fs["SleepHours"].mean(), 1)} hrs</b> average logged sleep
                </p>
                <div class="small-muted">
                These are descriptive statistics from the selected Fitbit records,
                not medical recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ACTIVITY EXPLORER

elif page == "🏃 Activity Explorer":

    st.subheader("🏃 Activity Explorer")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_kpi("Total Steps", fmt(fd.TotalSteps.sum()), "Aggregate", "green")
    with c2:
        render_kpi(
            "Total Distance",
            fmt(fd.TotalDistance.sum(), 1) + " km",
            "Recorded distance",
            "cyan",
        )
    with c3:
        render_kpi(
            "Total Active Minutes",
            fmt(fd.TotalActiveMinutes.sum()),
            "Light + moderate + very active",
            "amber",
        )
    with c4:
        render_kpi("Daily Records", fmt(len(fd)), "Rows", "purple")

    a, b = st.columns(2)

    with a:
        d = fd.groupby("Date", as_index=False).TotalSteps.mean()
        fig = px.area(
            d,
            x="Date",
            y="TotalSteps",
            title="Daily Steps Trend",
            color_discrete_sequence=["#22c55e"],
        )
        show_plot(fig, 330)

    with b:
        d = fd.groupby("Date", as_index=False).TotalDistance.mean()
        fig = px.line(
            d,
            x="Date",
            y="TotalDistance",
            markers=True,
            title="Daily Distance Covered",
            color_discrete_sequence=["#06b6d4"],
        )
        show_plot(fig, 330)

    st.subheader("📊 Activity Minutes by Weekday")

    activity_cols = [
        "LightlyActiveMinutes",
        "FairlyActiveMinutes",
        "VeryActiveMinutes",
    ]

    d = (
        fd.groupby("DayName", as_index=False)[activity_cols]
        .mean()
    )
    d["order"] = d["DayName"].map(WEEKDAY_ORDER)
    d = d.sort_values("order")

    fig = px.bar(
        d,
        x="DayName",
        y=activity_cols,
        barmode="group",
        title="Average Activity Categories by Weekday",
        color_discrete_sequence=["#06b6d4", "#f59e0b", "#f43f5e"],
    )
    show_plot(fig, 390)

    a, b = st.columns(2)

    with a:
        fig = px.scatter(
            fd,
            x="TotalSteps",
            y="Calories",
            size="TotalDistance",
            color="ActivityLevel",
            hover_data=["Id", "Date"],
            title="Steps vs Calories",
            color_discrete_map={
                "Low": "#f43f5e",
                "Moderate": "#f59e0b",
                "High": "#22c55e",
            },
        )
        show_plot(fig, 360)

    with b:
        fig = px.scatter(
            fd,
            x="TotalActiveMinutes",
            y="Calories",
            size="TotalSteps",
            color="ActivityLevel",
            hover_data=["Id", "Date"],
            title="Total Active Minutes vs Calories",
            color_discrete_map={
                "Low": "#f43f5e",
                "Moderate": "#f59e0b",
                "High": "#22c55e",
            },
        )
        show_plot(fig, 360)

    st.subheader("🏆 Top Activity Days")

    st.dataframe(
        fd.nlargest(20, "TotalSteps")[
            [
                "Id",
                "Date",
                "TotalSteps",
                "TotalDistance",
                "Calories",
                "TotalActiveMinutes",
                "VeryActiveMinutes",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

# PARTICIPANT ANALYSIS

elif page == "👥 Participant Analysis":

    st.subheader("👥 Participant-Level Analysis")
    st.caption(
        "The following views reproduce the participant × weekday analyses "
        "required in the case-study."
    )

    st.markdown("### 🚶 Average Steps — Participant × Weekday")

    step_heat = (
        daily.groupby(["Id", "DayName"])["TotalSteps"]
        .mean()
        .reset_index()
        .pivot(index="Id", columns="DayName", values="TotalSteps")
    )
    step_heat = step_heat.reindex(columns=WEEKDAYS)

    fig = px.imshow(
        step_heat,
        text_auto=".0f",
        aspect="auto",
        title="Average Steps by Participant and Weekday",
        color_continuous_scale="Greens",
    )
    show_plot(fig, 650)

    st.markdown("### 🪑 Average Sedentary Minutes — Participant × Weekday")

    sed_heat = (
        daily.groupby(["Id", "DayName"])["SedentaryMinutes"]
        .mean()
        .reset_index()
        .pivot(index="Id", columns="DayName", values="SedentaryMinutes")
    )
    sed_heat = sed_heat.reindex(columns=WEEKDAYS)

    fig = px.imshow(
        sed_heat,
        text_auto=".0f",
        aspect="auto",
        title="Average Sedentary Minutes by Participant and Weekday",
        color_continuous_scale="Oranges",
    )
    show_plot(fig, 650)

    st.markdown("### 🔥 Average Calories — Participant × Weekday")

    cal_heat = (
        daily.groupby(["Id", "DayName"])["Calories"]
        .mean()
        .reset_index()
        .pivot(index="Id", columns="DayName", values="Calories")
    )
    cal_heat = cal_heat.reindex(columns=WEEKDAYS)

    fig = px.imshow(
        cal_heat,
        text_auto=".0f",
        aspect="auto",
        title="Average Calories Burned by Participant and Weekday",
        color_continuous_scale="YlOrRd",
    )
    show_plot(fig, 650)

    a, b = st.columns(2)

    with a:
        top = (
            daily.groupby("Id", as_index=False)
            .agg(
                AvgSteps=("TotalSteps", "mean"),
                ActiveDays=("Date", "nunique"),
            )
            .nlargest(10, "AvgSteps")
        )

        fig = px.bar(
            top.sort_values("AvgSteps"),
            x="AvgSteps",
            y="Id",
            orientation="h",
            text_auto=".0f",
            title="Top 10 Participants by Average Steps",
            color_discrete_sequence=["#22c55e"],
        )
        show_plot(fig, 400)

    with b:
        top = (
            daily.groupby("Id", as_index=False)
            .agg(AvgSedentary=("SedentaryMinutes", "mean"))
            .nlargest(10, "AvgSedentary")
        )

        fig = px.bar(
            top.sort_values("AvgSedentary"),
            x="AvgSedentary",
            y="Id",
            orientation="h",
            text_auto=".0f",
            title="Top 10 Participants by Sedentary Minutes",
            color_discrete_sequence=["#f59e0b"],
        )
        show_plot(fig, 400)

# INTENSITY & CALORIES

elif page == "🔥 Intensity & Calories":

    st.subheader("🔥 Intensity & Calories")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_kpi("Calories", fmt(fd.Calories.sum()), "Total", "amber")
    with c2:
        render_kpi(
            "Avg Calories / Day",
            fmt(fd.Calories.mean()),
            "Daily average",
            "cyan",
        )
    with c3:
        render_kpi(
            "Active Minutes",
            fmt(fd.TotalActiveMinutes.mean(), 1),
            "Average/day",
            "green",
        )
    with c4:
        render_kpi(
            "Sedentary Minutes",
            fmt(fd.SedentaryMinutes.mean(), 1),
            "Average/day",
            "rose",
        )

    a, b = st.columns(2)

    with a:
        d = fd.groupby("Date", as_index=False).Calories.mean()
        fig = px.line(
            d,
            x="Date",
            y="Calories",
            markers=True,
            title="Daily Calories Burned",
            color_discrete_sequence=["#f59e0b"],
        )
        show_plot(fig, 330)

    with b:
        d = fd.groupby("Hour", as_index=False).Calories.mean() if "Hour" in fd.columns else (
            fh.groupby("Hour", as_index=False).Calories.mean()
        )
        fig = px.area(
            d,
            x="Hour",
            y="Calories",
            title="Calories Burned by Hour",
            color_discrete_sequence=["#06b6d4"],
        )
        show_plot(fig, 330)

    d = (
        fd.groupby("DayName", as_index=False)[
            ["LightlyActiveMinutes", "FairlyActiveMinutes", "VeryActiveMinutes"]
        ].mean()
    )
    d["order"] = d["DayName"].map(WEEKDAY_ORDER)
    d = d.sort_values("order")

    fig = px.bar(
        d,
        x="DayName",
        y=[
            "LightlyActiveMinutes",
            "FairlyActiveMinutes",
            "VeryActiveMinutes",
        ],
        barmode="stack",
        title="Activity Category Composition by Weekday",
        color_discrete_sequence=["#06b6d4", "#f59e0b", "#f43f5e"],
    )
    show_plot(fig, 400)

    a, b = st.columns(2)

    with a:
        fig = px.scatter(
            fd,
            x="TotalActiveMinutes",
            y="Calories",
            color="VeryActiveMinutes",
            size="TotalSteps",
            hover_data=["Id", "Date"],
            title="Calories vs Total Active Minutes",
            color_continuous_scale="Viridis",
        )
        show_plot(fig, 390)

    with b:
        d = fh.groupby("Hour", as_index=False).StepTotal.mean()
        fig = px.bar(
            d,
            x="Hour",
            y="StepTotal",
            title="Peak Active Hours",
            color_discrete_sequence=["#84cc16"],
        )
        show_plot(fig, 390)

# SLEEP

elif page == "😴 Sleep Analytics":

    st.subheader("😴 Sleep Analytics")

    if fs.empty:
        st.warning("No sleep records match the selected filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            render_kpi(
                "Average Sleep",
                fmt(fs.SleepHours.mean(), 1) + " hrs",
                "Logged average",
                "green",
            )
        with c2:
            render_kpi(
                "Time in Bed",
                fmt(fs.TimeInBedHours.mean(), 1) + " hrs",
                "Average",
                "cyan",
            )
        with c3:
            render_kpi(
                "Sleep Efficiency",
                fmt(fs.SleepEfficiencyPct.mean(), 1) + "%",
                "Recorded efficiency",
                "amber",
            )
        with c4:
            render_kpi(
                "Sleep Logs",
                fmt(len(fs)),
                "Records",
                "purple",
            )

        a, b = st.columns(2)

        with a:
            d = (
                fs.groupby("Date", as_index=False)
                [["SleepHours", "TimeInBedHours"]]
                .mean()
            )
            fig = px.line(
                d,
                x="Date",
                y=["SleepHours", "TimeInBedHours"],
                markers=True,
                title="Sleep Duration vs Time in Bed",
                color_discrete_sequence=["#22c55e", "#a855f7"],
            )
            show_plot(fig, 340)

        with b:
            fig = px.histogram(
                fs,
                x="SleepHours",
                nbins=20,
                title="Sleep Duration Distribution",
                color_discrete_sequence=["#06b6d4"],
            )
            show_plot(fig, 340)

        a, b = st.columns(2)

        with a:
            d = fs.groupby("DayName", as_index=False).SleepHours.mean()
            d["order"] = d["DayName"].map(WEEKDAY_ORDER)
            d = d.sort_values("order")

            fig = px.bar(
                d,
                x="DayName",
                y="SleepHours",
                title="Average Sleep by Weekday",
                text_auto=".1f",
                color_discrete_sequence=["#a855f7"],
            )
            show_plot(fig, 340)

        with b:
            merged = fd.merge(
                fs[["Id", "Date", "SleepHours"]],
                on=["Id", "Date"],
                how="inner",
            )

            if not merged.empty:
                fig = px.scatter(
                    merged,
                    x="SleepHours",
                    y="TotalSteps",
                    color="Calories",
                    hover_data=["Id", "Date"],
                    title="Sleep Duration vs Steps",
                    color_continuous_scale="Viridis",
                )
                show_plot(fig, 340)

        st.dataframe(
            fs[
                [
                    "Id",
                    "Date",
                    "TotalSleepRecords",
                    "SleepHours",
                    "TimeInBedHours",
                    "SleepEfficiencyPct",
                ]
            ].sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# HEART RATE

elif page == "❤️ Heart Rate":

    st.subheader("❤️ Heart Rate Analytics")

    if fhr.empty:
        st.warning("No heart-rate records match the selected filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            render_kpi(
                "Average HR",
                fmt(fhr.AvgHeartRate.mean(), 1) + " BPM",
                "Recorded average",
                "rose",
            )
        with c2:
            render_kpi(
                "Median HR",
                fmt(fhr.AvgHeartRate.median(), 1) + " BPM",
                "Median",
                "cyan",
            )
        with c3:
            render_kpi(
                "Min / Max",
                f"{fmt(fhr.MinHeartRate.min(),1)} / {fmt(fhr.MaxHeartRate.max(),1)}",
                "BPM",
                "amber",
            )
        with c4:
            render_kpi(
                "Data Points",
                fmt(len(fhr)),
                "Hourly readings",
                "purple",
            )

        d = (
            fhr.groupby("DateHour", as_index=False)
            .agg(AvgHeartRate=("AvgHeartRate", "mean"))
        )

        fig = px.line(
            d,
            x="DateHour",
            y="AvgHeartRate",
            title="Heart Rate Timeline",
            color_discrete_sequence=["#f43f5e"],
        )
        show_plot(fig, 380)

        a, b = st.columns(2)

        with a:
            d = (
                fhr.groupby("Hour", as_index=False)
                .agg(
                    AvgHR=("AvgHeartRate", "mean"),
                    MinHR=("MinHeartRate", "mean"),
                    MaxHR=("MaxHeartRate", "mean"),
                )
            )

            fig = px.line(
                d,
                x="Hour",
                y=["AvgHR", "MinHR", "MaxHR"],
                markers=True,
                title="Heart Rate by Hour",
                color_discrete_sequence=["#f43f5e", "#06b6d4", "#f59e0b"],
            )
            show_plot(fig, 350)

        with b:
            fig = px.histogram(
                fhr,
                x="AvgHeartRate",
                nbins=30,
                title="Average Heart Rate Distribution",
                color_discrete_sequence=["#f43f5e"],
            )
            show_plot(fig, 350)

# WEIGHT / BMI

elif page == "⚖️ Weight & BMI":

    st.subheader("⚖️ Weight & BMI")

    if fw.empty:
        st.warning("No weight records match the selected filters.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            render_kpi(
                "Average Weight",
                fmt(fw.WeightKg.mean(), 1) + " kg",
                "Recorded",
                "green",
            )
        with c2:
            render_kpi(
                "Average BMI",
                fmt(fw.BMI.mean(), 1),
                "Recorded BMI",
                "cyan",
            )
        with c3:
            render_kpi(
                "Weight Logs",
                fmt(len(fw)),
                "Records",
                "purple",
            )

        selected_weight_id = st.selectbox(
            "Focus on one participant",
            ["All"] + sorted(fw.Id.astype(str).unique().tolist()),
        )

        chart_df = fw.copy()
        if selected_weight_id != "All":
            chart_df = chart_df[
                chart_df["Id"].astype(str) == selected_weight_id
            ]

        a, b = st.columns(2)

        with a:
            fig = px.line(
                chart_df.sort_values("DateOnly"),
                x="DateOnly",
                y="WeightKg",
                color="Id" if selected_weight_id == "All" else None,
                markers=True,
                title="Weight Trend Over Time",
            )
            show_plot(fig, 360)

        with b:
            bmi_df = chart_df.dropna(subset=["BMI"])
            if not bmi_df.empty:
                fig = px.line(
                    bmi_df.sort_values("DateOnly"),
                    x="DateOnly",
                    y="BMI",
                    color="Id" if selected_weight_id == "All" else None,
                    markers=True,
                    title="BMI Trend Over Time",
                )
                show_plot(fig, 360)

# DATA QUALITY

elif page == "🧹 Data Quality":

    st.subheader("🧹 Data Quality & Cleaning Report")

    st.caption(
        "This page documents the quality of the cleaned analytics tables "
        "currently loaded by the Streamlit application."
    )

    tables = {
        "Daily Activity": daily,
        "Hourly Activity": hourly,
        "Sleep": sleep,
        "Heart Rate Hourly": hr,
        "Weight": weight,
        "Participant Summary": participants,
    }

    quality_rows = []

    for name, df in tables.items():
        quality_rows.append(
            {
                "Dataset": name,
                "Rows": len(df),
                "Columns": len(df.columns),
                "Missing Cells": int(df.isna().sum().sum()),
                "Duplicate Rows": int(df.duplicated().sum()),
            }
        )

    quality_df = pd.DataFrame(quality_rows)

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_kpi(
            "Participants",
            fmt(daily.Id.nunique()),
            "Unique IDs",
            "green",
        )
    with c2:
        render_kpi(
            "Activity Records",
            fmt(len(daily)),
            "Daily rows",
            "cyan",
        )
    with c3:
        render_kpi(
            "Sleep Records",
            fmt(len(sleep)),
            "Sleep rows",
            "purple",
        )
    with c4:
        render_kpi(
            "HR Records",
            fmt(len(hr)),
            "Hourly rows",
            "rose",
        )

    st.markdown("### 📅 Data Coverage")

    coverage = []

    for name, df, col in [
        ("Daily Activity", daily, "Date"),
        ("Sleep", sleep, "Date"),
        ("Hourly Activity", hourly, "DateHour"),
        ("Heart Rate", hr, "DateHour"),
        ("Weight", weight, "DateOnly"),
    ]:
        if col in df.columns:
            coverage.append(
                {
                    "Dataset": name,
                    "Start": df[col].min(),
                    "End": df[col].max(),
                }
            )

    st.dataframe(
        pd.DataFrame(coverage),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 🧽 Cleaning / Preparation Checklist")

    st.markdown(
        """
        - ✅ Dates and timestamps standardized.
        - ✅ Numeric fields loaded as numeric values.
        - ✅ Duplicate records checked.
        - ✅ Missing values profiled for each analytics table.
        - ✅ Heart-rate data is represented at hourly level for dashboard performance.
        - ✅ Sleep data is represented at daily-record level.
        - ✅ Total Active Minutes is derived from light + fairly + very active minutes.
        - ✅ Activity-level categories are available for filtering and comparison.
        - ✅ Participant-level summary tables are available for SQL analysis.
        """
    )

    st.markdown("### ⚠️ Dataset Limitations")

    st.markdown(
        """
        - The dataset represents a limited set of Fitbit participants rather than
          the entire population.
        - Participants do not necessarily have the same number of recorded days.
        - Different Fitbit metrics have different levels of data coverage.
        - Weight records are much fewer than daily activity records.
        - These records describe observed device usage; they should not be treated
          as clinical or medical measurements.
        """
    )

# SQL INSIGHTS

elif page == "🧠 SQL Insights":

    st.subheader("🧠 SQL Analysis & Database Insights")

    st.caption(
        "All preset analyses below are executed directly against the SQLite database."
    )

    qmap = {
        "Overall KPIs": """
            SELECT
                COUNT(*) AS activity_rows,
                COUNT(DISTINCT Id) AS participants,
                ROUND(AVG(TotalSteps),2) AS avg_steps,
                ROUND(AVG(Calories),2) AS avg_calories,
                ROUND(AVG(TotalDistance),2) AS avg_distance,
                ROUND(AVG(SedentaryMinutes),2) AS avg_sedentary_minutes,
                ROUND(AVG(VeryActiveMinutes),2) AS avg_very_active_minutes
            FROM daily_activity_clean
        """,

        "Average Steps by Weekday": """
            SELECT
                DayName,
                ROUND(AVG(TotalSteps),2) AS avg_steps,
                COUNT(*) AS records
            FROM daily_activity_clean
            GROUP BY DayOfWeek, DayName
            ORDER BY DayOfWeek
        """,

        "Participant Steps by Weekday": """
            SELECT
                Id,
                DayName,
                ROUND(AVG(TotalSteps),2) AS avg_steps
            FROM daily_activity_clean
            GROUP BY Id, DayName
            ORDER BY Id, DayOfWeek
        """,

        "Participant Sedentary Minutes by Weekday": """
            SELECT
                Id,
                DayName,
                ROUND(AVG(SedentaryMinutes),2) AS avg_sedentary_minutes
            FROM daily_activity_clean
            GROUP BY Id, DayName
            ORDER BY Id, DayOfWeek
        """,

        "Participant Calories by Weekday": """
            SELECT
                Id,
                DayName,
                ROUND(AVG(Calories),2) AS avg_calories
            FROM daily_activity_clean
            GROUP BY Id, DayName
            ORDER BY Id, DayOfWeek
        """,

        "Activity Categories by Weekday": """
            SELECT
                DayName,
                ROUND(AVG(LightlyActiveMinutes),2) AS avg_light_minutes,
                ROUND(AVG(FairlyActiveMinutes),2) AS avg_fair_minutes,
                ROUND(AVG(VeryActiveMinutes),2) AS avg_very_active_minutes
            FROM daily_activity_clean
            GROUP BY DayOfWeek, DayName
            ORDER BY DayOfWeek
        """,

        "Calories vs Total Active Minutes": """
            SELECT
                Id,
                Date,
                TotalSteps,
                Calories,
                (VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes)
                    AS TotalActiveMinutes
            FROM daily_activity_clean
            ORDER BY TotalActiveMinutes DESC
            LIMIT 100
        """,

        "Calories Burned by Hour": """
            SELECT
                Hour,
                ROUND(AVG(Calories),2) AS avg_calories,
                ROUND(AVG(StepTotal),2) AS avg_steps
            FROM hourly_activity_clean
            GROUP BY Hour
            ORDER BY Hour
        """,

        "Top 10 Active Participants": """
            SELECT
                Id,
                ActiveDays,
                ROUND(AvgSteps,2) AS AvgSteps,
                ROUND(AvgCalories,2) AS AvgCalories
            FROM participant_summary
            ORDER BY AvgSteps DESC
            LIMIT 10
        """,

        "Top 10 Sedentary Participants": """
            SELECT
                Id,
                ActiveDays,
                ROUND(AvgSedentaryMinutes,2) AS AvgSedentaryMinutes
            FROM participant_summary
            ORDER BY AvgSedentaryMinutes DESC
            LIMIT 10
        """,

        "Sleep vs Activity": """
            SELECT
                d.Id,
                ROUND(AVG(d.TotalSteps),2) AS avg_steps,
                ROUND(AVG(d.Calories),2) AS avg_calories,
                ROUND(AVG(d.VeryActiveMinutes),2) AS avg_very_active_minutes,
                ROUND(AVG(s.SleepHours),2) AS avg_sleep_hours,
                ROUND(AVG(s.SleepEfficiencyPct),2) AS avg_sleep_efficiency
            FROM daily_activity_clean d
            JOIN sleep_clean s
              ON d.Id = s.Id
             AND d.Date = s.Date
            GROUP BY d.Id
        """,
    }

    choice = st.selectbox(
        "Choose a SQL Analysis",
        list(qmap.keys()),
    )

    result = sql_df(qmap[choice])

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("### 🔎 Run Custom SELECT SQL")

    sql = st.text_area(
        "SQL Query",
        """
SELECT
    Id,
    Date,
    TotalSteps,
    Calories,
    SedentaryMinutes
FROM daily_activity_clean
ORDER BY TotalSteps DESC
LIMIT 20;
""".strip(),
        height=150,
    )

    if st.button("▶ Execute SQL", type="primary"):
        clean_sql = sql.strip().lower()

        if not (
            clean_sql.startswith("select")
            or clean_sql.startswith("with")
        ):
            st.error("Only read-only SELECT / WITH queries are allowed.")
        else:
            try:
                result = pd.read_sql_query(sql, get_connection())
                st.success(f"Query executed successfully — {len(result):,} rows.")
                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as exc:
                st.error(f"SQL Error: {exc}")

# PYTHON EDA — MATPLOTLIB + SEABORN

elif page == "📊 Python EDA":

    st.subheader("📊 Python EDA — Matplotlib & Seaborn")

    st.caption(
        "This section complements the interactive Plotly dashboard with "
        "static exploratory-analysis visuals using Matplotlib and Seaborn."
    )

    chart = st.selectbox(
        "Choose EDA Visualization",
        [
            "Correlation Heatmap",
            "Steps by Weekday",
            "Calories Distribution",
            "Activity Minutes Distribution",
            "Participant Steps Heatmap",
            "Sleep Distribution",
        ],
    )

    plt.figure(figsize=(12, 5))

    if chart == "Correlation Heatmap":
        cols = [
            c for c in [
                "TotalSteps",
                "Calories",
                "TotalDistance",
                "VeryActiveMinutes",
                "FairlyActiveMinutes",
                "LightlyActiveMinutes",
                "SedentaryMinutes",
            ]
            if c in daily.columns
        ]

        sns.heatmap(
            daily[cols].corr(),
            annot=True,
            fmt=".2f",
            cmap="Greens",
            linewidths=.5,
        )
        plt.title("Fitbit Activity Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    elif chart == "Steps by Weekday":
        d = (
            daily.groupby("DayName")["TotalSteps"]
            .mean()
            .reindex(WEEKDAYS)
        )

        sns.barplot(
            x=d.index,
            y=d.values,
        )
        plt.title("Average Steps by Weekday")
        plt.xlabel("Day")
        plt.ylabel("Average Steps")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    elif chart == "Calories Distribution":
        sns.histplot(
            daily["Calories"].dropna(),
            bins=30,
            kde=True,
        )
        plt.title("Daily Calories Distribution")
        plt.xlabel("Calories")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    elif chart == "Activity Minutes Distribution":
        cols = [
            "LightlyActiveMinutes",
            "FairlyActiveMinutes",
            "VeryActiveMinutes",
        ]
        sns.boxplot(data=daily[cols])
        plt.title("Distribution of Activity Minutes")
        plt.ylabel("Minutes")
        plt.xticks(
            range(len(cols)),
            ["Light", "Fairly Active", "Very Active"],
        )
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    elif chart == "Participant Steps Heatmap":
        heat = (
            daily.groupby(["Id", "DayName"])["TotalSteps"]
            .mean()
            .reset_index()
            .pivot(index="Id", columns="DayName", values="TotalSteps")
            .reindex(columns=WEEKDAYS)
        )

        sns.heatmap(
            heat,
            cmap="Greens",
            linewidths=.2,
        )
        plt.title("Participant × Weekday Average Steps")
        plt.xlabel("Weekday")
        plt.ylabel("Participant")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    elif chart == "Sleep Distribution":
        sns.histplot(
            sleep["SleepHours"].dropna(),
            bins=20,
            kde=True,
        )
        plt.title("Sleep Duration Distribution")
        plt.xlabel("Sleep Hours")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

# INSIGHTS & RECOMMENDATIONS

elif page == "💡 Insights & Recommendations":

    st.subheader("💡 Key Findings & Product Opportunities")

    avg_steps = daily["TotalSteps"].mean()
    avg_sedentary = daily["SedentaryMinutes"].mean()
    avg_active = daily["TotalActiveMinutes"].mean()
    avg_sleep = sleep["SleepHours"].mean()
    avg_eff = sleep["SleepEfficiencyPct"].mean()
    avg_calories = daily["Calories"].mean()

    weekday_steps = (
        daily.groupby("DayName")["TotalSteps"]
        .mean()
        .reindex(WEEKDAYS)
    )

    best_step_day = weekday_steps.idxmax()
    lowest_step_day = weekday_steps.idxmin()

    hourly_steps = hourly.groupby("Hour")["StepTotal"].mean()
    peak_hour = hourly_steps.idxmax()

    st.markdown("### 📌 Data-Driven Findings")

    findings = [
        (
            "🚶 Activity",
            f"Average recorded activity was {fmt(avg_steps)} steps per day."
        ),
        (
            "🪑 Sedentary Behaviour",
            f"Average sedentary time was {fmt(avg_sedentary,1)} minutes per day."
        ),
        (
            "🔥 Active Time",
            f"Average total active time was {fmt(avg_active,1)} minutes per day."
        ),
        (
            "🔥 Calories",
            f"Average recorded calorie expenditure was {fmt(avg_calories)} calories per day."
        ),
        (
            "😴 Sleep",
            f"Average logged sleep was {fmt(avg_sleep,1)} hours with "
            f"{fmt(avg_eff,1)}% average sleep efficiency."
        ),
        (
            "📅 Weekday Pattern",
            f"{best_step_day} had the highest average recorded steps, while "
            f"{lowest_step_day} had the lowest."
        ),
        (
            "⏰ Hourly Pattern",
            f"The highest average hourly step volume occurred around hour {int(peak_hour)}."
        ),
    ]

    for title, text in findings:
        st.markdown(
            f"""
            <div class="section-card" style="margin-bottom:10px;">
                <b>{title}</b>
                <div class="small-muted" style="margin-top:5px;">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💼 Product / Marketing Opportunities")

    recommendations = [
        (
            "1. Personalized activity summaries",
            "Use participant-level and weekday patterns to provide weekly "
            "summaries instead of only showing raw totals."
        ),
        (
            "2. Inactivity awareness",
            "Because sedentary minutes are substantial in the dataset, a product "
            "could surface inactivity periods and weekly sedentary trends."
        ),
        (
            "3. Activity-category tracking",
            "Separate light, fairly active and very active minutes so users can "
            "understand how their total activity is composed."
        ),
        (
            "4. Sleep + activity dashboard",
            "Present sleep duration and activity trends together so users can "
            "explore their own recorded patterns."
        ),
        (
            "5. Time-of-day insights",
            "Use hourly activity patterns to show when recorded movement tends "
            "to be highest and help users review their routines."
        ),
    ]

    for title, text in recommendations:
        st.markdown(
            f"""
            <div class="insight-card" style="margin-bottom:10px;">
                <div class="insight-title">{title}</div>
                <div>{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ⚠️ Interpretation Note")
    st.info(
        "These findings are descriptive observations from the Fitbit dataset. "
        "They are not clinical conclusions and do not establish causation."
    )

    st.markdown("### 📋 Case-Study Submission Checklist")

    checklist = pd.DataFrame(
        {
            "Requirement": [
                "Data cleaning / preparation",
                "SQL analysis",
                "Interactive Streamlit dashboard",
                "Participant analysis",
                "Activity-category analysis",
                "Calories vs active minutes",
                "Sleep analysis",
                "Heart-rate analysis",
                "Weight / BMI analysis",
                "Matplotlib / Seaborn EDA",
                "Key findings",
                "Recommendations",
                "Data limitations",
                "Video recording",
                "Power BI / Tableau deliverable",
            ],
            "Status": [
                "✅ Included",
                "✅ Included",
                "✅ Included",
                "✅ Included",
                "✅ Included",
                "✅ Included",
                "✅ Included",
                "✅ Included",
                "✅ Included",
                "✅ Included in this app",
                "✅ Included",
                "✅ Included",
                "✅ Included",
                "🎥 Record separately",
                "📊 Prepare separately if required",
            ],
        }
    )

    st.dataframe(
        checklist,
        use_container_width=True,
        hide_index=True,
    )

# DATA EXPLORER

elif page == "🔎 Data Explorer":

    st.subheader("🔎 Data Explorer")

    maps = {
        "Daily Activity": daily,
        "Hourly Activity": hourly,
        "Sleep": sleep,
        "Heart Rate Hourly": hr,
        "Weight": weight,
        "Participant Summary": participants,
    }

    name = st.selectbox("Choose Table", list(maps))
    df = maps[name].copy()

    search = st.text_input("🔍 Search across table")

    if search:
        mask = (
            df.astype(str)
            .apply(
                lambda col: col.str.contains(
                    search,
                    case=False,
                    na=False,
                )
            )
            .any(axis=1)
        )
        df = df[mask]

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        with st.expander("📈 Quick Statistics"):
            st.dataframe(
                df[numeric_cols].describe().T,
                use_container_width=True,
            )

    st.caption(
        f"Showing {len(df):,} records × {len(df.columns):,} columns"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="fitbit_filtered_data.csv",
        mime="text/csv",
    )
