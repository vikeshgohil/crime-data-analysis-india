import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis import (
    load_data,
    get_city_options,
    filter_data,
    compute_kpis,
    top_cities,
    trend_over_time,
    crime_type_distribution,
    gender_breakdown,
    heatmap_pivot,
    linear_regression_predict
)

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Crime Data Analysis of Indian Cities", layout="wide")

# ------------------ Simple Login Page ------------------
def login_page():
    st.markdown(
        """
        <h2 style='text-align:center; color:#1E88E5;'>🔐 Login to Crime Data Dashboard</h2>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.session_state["authenticated"] = True
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid username or password. Try again.")

# ------------------ Logout Button ------------------
def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.success("Logged out successfully!")
        st.rerun()

# ------------------ Main Dashboard ------------------
def main_dashboard():
    st.title("Crime Data Analysis of Indian Cities")
    st.caption("Mini Project • Data Analytics & Visualization • Streamlit Dashboard")

    # Sidebar: Data Source and Filters
    st.sidebar.header("📂 Data & Filters")
    logout_button()

    data_path = st.sidebar.text_input(
        "Data file path",
        "data/sample_crime_data.csv",
        help="CSV or Excel file path. Default uses the included sample dataset."
    )

    df = load_data(data_path)
    if df.empty:
        st.stop()

    # Filter widgets
    years = sorted([int(y) for y in df["year"].dropna().unique().tolist()])
    states = sorted(df["state_ut"].dropna().unique().tolist())
    crimes = sorted(df["crime_type"].dropna().unique().tolist())

    sel_years = st.sidebar.multiselect("Year(s)", years, default=years)
    sel_states = st.sidebar.multiselect("State/UT", states)
    city_options = get_city_options(df, sel_states)
    sel_cities = st.sidebar.multiselect("City", city_options)
    sel_crimes = st.sidebar.multiselect("Crime Type", crimes)

    fdf = filter_data(df, sel_years, sel_states, sel_cities, sel_crimes)

    if fdf.empty:
        st.warning("No data after applying filters. Try changing your selections.")
        st.stop()

    # ------------------ KPIs ------------------
    kpis = compute_kpis(fdf)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Cases", f"{kpis['total_cases']:,}")
    k2.metric("Cities in View", f"{kpis['n_cities']}")
    yr_label = f"{min(kpis['years'])} – {max(kpis['years'])}" if kpis['years'] else "N/A"
    k3.metric("Year Range", yr_label)
    k4.metric("Top Crime Type", kpis['top_crime'] if kpis['top_crime'] else "N/A")

    st.markdown("---")

    # ------------------ Row 1: Trend & Crime Type Distribution ------------------
    c1, c2 = st.columns([2, 1])
    with c1:
        trend = trend_over_time(fdf)
        fig1 = px.line(trend, x="year", y="total_cases", markers=True, title="Trend of Total Crimes Over Time")
        fig1.update_layout(height=420)
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        ct = crime_type_distribution(fdf)
        fig2 = px.bar(ct, x="total_cases", y="crime_type", orientation="h", title="Crime Type Distribution")
        fig2.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    # ------------------ Row 2: Top Cities & Gender Breakdown ------------------
    c3, c4 = st.columns(2)
    with c3:
        top = top_cities(fdf, n=10)
        fig3 = px.bar(top, x="city", y="total_cases", title="Top 10 Cities by Total Cases")
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        gb = gender_breakdown(fdf)
        if not gb.empty:
            fig4 = go.Figure()
            fig4.add_bar(x=gb["crime_type"], y=gb.get("victims_male", 0), name="Victims Male")
            fig4.add_bar(x=gb["crime_type"], y=gb.get("victims_female", 0), name="Victims Female")
            fig4.update_layout(barmode="stack", title="Victims by Gender (Stacked)")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Gender columns not found in this dataset.")

    # ------------------ Row 3: Heatmap ------------------
    piv = heatmap_pivot(fdf, index="year", column="crime_type")
    if not piv.empty:
        fig5 = px.imshow(piv, aspect="auto", title="Heatmap: Year vs Crime Type (Total Cases)")
        st.plotly_chart(fig5, use_container_width=True)

    # ------------------ Linear Regression Prediction ------------------
    st.markdown("---")
    st.subheader("🤖 Predict Crime Trend Using Linear Regression")

    numeric_cols = fdf.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(numeric_cols) >= 2:
        x_feature = st.selectbox("Select Feature (X)", numeric_cols, key="lr_x")
        y_feature = st.selectbox("Select Target (Y)", numeric_cols, key="lr_y")
        future_value = st.number_input(f"Enter {x_feature} value to predict {y_feature}", min_value=0.0)

        if st.button("Predict Crime", key="lr_button"):
            prediction = linear_regression_predict(fdf, x_feature, y_feature, future_value)
            st.success(f"Predicted {y_feature}: {prediction:.0f}")
    else:
        st.info("Not enough numeric columns for Linear Regression prediction.")

    # ------------------ Data Preview ------------------
    with st.expander("📋 Data Preview"):
        st.dataframe(fdf.head(100))

    st.success("Dashboard ready.")
    st.caption(
        "The app automatically standardizes columns (State/UT, City, Year, Crime_Type, Total_Cases, Victims_Male, Victims_Female)."
    )


# ------------------ Entry Point ------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_page()
else:
    main_dashboard()
