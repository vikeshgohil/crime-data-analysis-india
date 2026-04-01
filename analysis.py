import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.linear_model import LinearRegression

# ------------------ Helper: Standardize Columns ------------------
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase + replace spaces, slashes, and hyphens
    col_map = {c: c.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_") for c in df.columns}
    df = df.rename(columns=col_map)

    # Flexible alias map for column name variations
    aliases = {
        "state_ut": ["state_ut", "state", "state/ut", "state___ut"],
        "city": ["city", "district", "district_city"],
        "year": ["year"],
        "crime_type": ["crime_type", "crime_head", "offence", "crime"],
        "total_cases": ["total_cases", "total", "cases", "count"],
        "victims_male": ["victims_male", "male_victims", "male"],
        "victims_female": ["victims_female", "female_victims", "female"]
    }

    for target, opts in aliases.items():
        for opt in opts:
            if opt in df.columns:
                df = df.rename(columns={opt: target})
                break

    # Ensure required columns exist
    required = ["state_ut", "city", "year", "crime_type", "total_cases"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # Convert to numeric types
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    for col in ["total_cases", "victims_male", "victims_female"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Clean strings (remove spaces)
    for col in ["state_ut", "city", "crime_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


# ------------------ Load Data ------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """
    Loads CSV or Excel data and standardizes column names.
    """
    if not os.path.exists(path):
        st.error(f"Data file not found: {path}")
        return pd.DataFrame()

    ext = os.path.splitext(path)[1].lower()
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df = _standardize_columns(df)
    # Drop invalid rows
    df = df.dropna(subset=["year", "city"])
    return df


# ------------------ Dropdown Options ------------------
def get_city_options(df: pd.DataFrame, selected_states):
    if selected_states:
        return sorted(df[df["state_ut"].isin(selected_states)]["city"].unique().tolist())
    return sorted(df["city"].unique().tolist())


# ------------------ Data Filtering ------------------
def filter_data(df: pd.DataFrame, years, states, cities, crimes):
    out = df.copy()
    if years:
        out = out[out["year"].isin(years)]
    if states:
        out = out[out["state_ut"].isin(states)]
    if cities:
        out = out[out["city"].isin(cities)]
    if crimes:
        out = out[out["crime_type"].isin(crimes)]
    return out


# ------------------ KPI Calculations ------------------
def compute_kpis(df: pd.DataFrame):
    total = int(df.get("total_cases", pd.Series(dtype=int)).sum()) if not df.empty else 0
    n_cities = df["city"].nunique() if "city" in df.columns else 0
    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []
    top_crime = None
    if not df.empty and "crime_type" in df.columns:
        s = df.groupby("crime_type")["total_cases"].sum().sort_values(ascending=False)
        if len(s) > 0:
            top_crime = s.index[0]
    return {
        "total_cases": total,
        "n_cities": n_cities,
        "years": years,
        "top_crime": top_crime
    }


# ------------------ Group Sum ------------------
def group_sum(df: pd.DataFrame, by_cols):
    if df.empty:
        return pd.DataFrame(columns=by_cols + ["total_cases"])
    return df.groupby(by_cols, as_index=False)["total_cases"].sum()


# ------------------ Charts Data Functions ------------------
def top_cities(df: pd.DataFrame, n=10):
    g = group_sum(df, ["city"])
    return g.sort_values("total_cases", ascending=False).head(n)


def trend_over_time(df: pd.DataFrame):
    return group_sum(df, ["year"]).sort_values("year")


def crime_type_distribution(df: pd.DataFrame):
    return group_sum(df, ["crime_type"]).sort_values("total_cases", ascending=False)


def gender_breakdown(df: pd.DataFrame):
    out = pd.DataFrame(columns=["crime_type", "victims_male", "victims_female"])
    if df.empty:
        return out
    cols = [c for c in ["victims_male", "victims_female"] if c in df.columns]
    if not cols:
        return out
    agg = df.groupby("crime_type")[cols].sum().reset_index()
    return agg


def heatmap_pivot(df: pd.DataFrame, index="year", column="crime_type"):
    if df.empty:
        return pd.DataFrame()
    piv = df.pivot_table(values="total_cases", index=index, columns=column, aggfunc="sum", fill_value=0)
    return piv


# ------------------ Linear Regression (ML) ------------------
def linear_regression_predict(df: pd.DataFrame, x_col: str, y_col: str, future_x: float) -> float:
    """
    Train a simple Linear Regression model and predict Y for given X.
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return 0.0

    X = df[[x_col]].values
    y = df[y_col].values

    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict([[future_x]])
    return float(pred[0])
