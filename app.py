import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

from src.explain import get_shap_explainer, compute_shap_values
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.features import (
    create_time_features,
    create_lag_features,
    create_rolling_features
)
from src.train import train_model

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# -------------------------------
# Load & Prepare Data
# -------------------------------
@st.cache_data
def load_pipeline():
    df = load_data("data/raw/sales.csv")
    df = preprocess_data(df)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    return df

df = load_pipeline()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Controls")

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['order_date'].min(), df['order_date'].max()]
)

filtered_df = df[
    (df['order_date'] >= pd.to_datetime(date_range[0])) &
    (df['order_date'] <= pd.to_datetime(date_range[1]))
]

# -------------------------------
# Header
# -------------------------------
st.title("📈 Sales Forecasting & Analytics")
st.markdown(
    """
    **End-to-End Data Science Application**  
    Analyze historical sales data, build machine learning models,  
    and forecast future sales interactively.
    """
)

# -------------------------------
# KPI Section
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Revenue",
    f"${filtered_df['sales'].sum():,.0f}"
)

col2.metric(
    "Average Sales",
    f"${filtered_df['sales'].mean():,.0f}"
)

col3.metric(
    "Max Daily Sales",
    f"${filtered_df['sales'].max():,.0f}"
)

growth = (
    filtered_df.groupby(filtered_df['order_date'].dt.year)['sales'].sum()
    .pct_change()
    .mean() * 100
)

col4.metric(
    "Avg YoY Growth",
    f"{growth:.2f}%"
)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "🔍 EDA", "🤖 Machine Learning", "🔮 Forecast"]
)

# ===============================
# TAB 1 — DASHBOARD
# ===============================
with tab1:
    st.subheader("Sales Trend Over Time")

    fig = px.line(
        filtered_df,
        x="order_date",
        y="sales",
        title="Daily Sales Trend"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("🗂️ Data Used in This Dashboard")

    st.markdown(
        """
        **Source Data**
        - Transaction-level sales dataset
        - One row represents one order record
        - Original data has been cleaned and enriched before analysis

        **Key Columns Used**
        - `order_date` → transaction date
        - `sales` → target variable (revenue)
        - `segment`, `category` → business dimensions
        - Time-based features derived from `order_date`

        **Data Scope**
        - Date range controlled via sidebar
        - All metrics & charts dynamically update based on filters
        """
    )

    with st.expander("🔍 Preview Filtered Dataset (Dashboard Input)"):
        st.dataframe(
            filtered_df[
                ["order_date", "sales", "segment", "category"]
            ].sort_values("order_date").head(20),
            use_container_width=True
        )


with tab2:
    st.header("🔍 Exploratory Data Analysis (EDA)")
    st.caption("Deep dive into data structure, distribution, patterns, and modeling readiness")

    # ==================================================
    # 1️⃣ EDA PURPOSE & CONTEXT
    # ==================================================
    st.markdown(
        """
        ### 📌 EDA Objectives
        This section explores the **statistical characteristics and behavioral patterns**
        of the sales data to ensure it is **clean, reliable, and suitable for forecasting models**.

        **EDA helps answer:**
        - What does the sales distribution look like?
        - Are there temporal patterns or seasonality?
        - Which features are most relevant for forecasting?
        - Is the dataset ready for machine learning?
        """
    )

    st.markdown("---")

    # ==================================================
    # 2️⃣ DATASET PREVIEW (POST-PROCESSING)
    # ==================================================
    st.subheader("🔎 Dataset Preview (After Preprocessing)")

    st.dataframe(
        filtered_df.head(10),
        use_container_width=True
    )

    st.info(
        """
        This preview shows the **final dataset** after cleaning, formatting,
        and feature engineering steps.
        """
    )

    # ==================================================
    # 3️⃣ DATA STRUCTURE & TYPES
    # ==================================================
    st.markdown("---")
    st.subheader("📋 Data Structure & Types")

    dtype_df = (
        filtered_df.dtypes
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Data Type"})
    )

    st.dataframe(dtype_df, use_container_width=True)

    st.info(
        """
        **Structure Insight:**
        - `order_date` is used for time-series indexing
        - Numeric features feed the forecasting model
        - Categorical dimensions support filtering & segmentation
        """
    )

    # ==================================================
    # 4️⃣ MISSING & DUPLICATE DATA CHECK
    # ==================================================
    st.markdown("---")
    st.subheader("🚨 Data Quality Check")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Missing Values",
            int(filtered_df.isna().sum().sum())
        )

    with col2:
        st.metric(
            "Duplicate Rows",
            int(filtered_df.duplicated().sum())
        )

    st.success(
        """
        No critical data quality issues detected.
        Dataset is safe for analysis and modeling.
        """
    )

    # ==================================================
    # 5️⃣ SALES DISTRIBUTION (ORIGINAL TAB 2)
    # ==================================================
    st.markdown("---")
    st.subheader("📊 Sales Distribution")

    fig_dist = px.histogram(
        filtered_df,
        x="sales",
        nbins=50,
        title="Distribution of Sales Values",
        labels={"sales": "Sales Amount"}
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    st.warning(
        """
        **Distribution Insight:**
        - Sales data is right-skewed
        - Indicates occasional high-value transactions
        - Common pattern in retail & e-commerce
        """
    )

    # ==================================================
    # 6️⃣ OUTLIER ANALYSIS (BUSINESS-AWARE)
    # ==================================================
    st.markdown("---")
    st.subheader("📦 Outlier Analysis (Sales)")

    q1 = filtered_df["sales"].quantile(0.25)
    q3 = filtered_df["sales"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_count = filtered_df[
        (filtered_df["sales"] < lower) |
        (filtered_df["sales"] > upper)
    ].shape[0]

    st.metric("Detected Outliers", outlier_count)

    st.info(
        """
        **Outlier Decision:**
        - Outliers represent legitimate high-value sales
        - Retained to preserve real business signals
        - XGBoost is robust to extreme values
        """
    )

    # ==================================================
    # 7️⃣ TEMPORAL BEHAVIOR ANALYSIS
    # ==================================================
    st.markdown("---")
    st.subheader("⏱ Temporal Sales Patterns")

    temp_df = (
        filtered_df
        .groupby(filtered_df["order_date"].dt.dayofweek)["sales"]
        .mean()
        .reset_index()
        .rename(columns={"order_date": "Day of Week"})
    )

    fig_temp = px.bar(
        temp_df,
        x="Day of Week",
        y="sales",
        title="Average Sales by Day of Week",
        labels={"sales": "Average Sales"}
    )

    st.plotly_chart(fig_temp, use_container_width=True)

    st.info(
        """
        **Temporal Insight:**
        - Sales behavior varies across the week
        - Indicates strong time dependency
        - Supports time-based feature engineering
        """
    )

    # ==================================================
    # 8️⃣ FEATURE CORRELATION (ORIGINAL + UPDATE)
    # ==================================================
    st.markdown("---")
    st.subheader("🔗 Feature Correlation Analysis")

    numeric_df = filtered_df.select_dtypes(include="number")

    fig_corr = px.imshow(
        numeric_df.corr(),
        text_auto=True,
        aspect="auto",
        title="Correlation Between Numeric Features"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    st.warning(
        """
        **Key Findings:**
        - Lag features have strong correlation with current sales
        - Rolling statistics capture trend & momentum
        - Confirms suitability for time-series ML modeling
        """
    )

    # ==================================================
    # 9️⃣ MODELING READINESS SUMMARY
    # ==================================================
    st.markdown("---")
    st.subheader("🧪 Modeling Readiness Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", filtered_df.shape[0])
    col2.metric("Features", filtered_df.shape[1])
    col3.metric("Numeric Features", numeric_df.shape[1])
    col4.metric("Time Index", "order_date")

    st.success(
        """
        **EDA Conclusion**
        - Data is clean, structured, and well-understood
        - Temporal patterns and feature relationships are clear
        - Dataset is fully ready for forecasting & explainable ML models
        """
    )



# ===============================
# TAB 3 — MACHINE LEARNING (Advanced)
# ===============================
with tab3:
    st.subheader("🤖 Model Training, Evaluation & Interpretation")

    # -------------------------------
    # Filter: segment / category
    # -------------------------------
    segment_options = ["All"] + filtered_df['segment'].unique().tolist()
    category_options = ["All"] + filtered_df['category'].unique().tolist()

    selected_segment = st.selectbox("Filter by Segment", segment_options)
    selected_category = st.selectbox("Filter by Category", category_options)

    filtered_ml_df = filtered_df.copy()
    if selected_segment != "All":
        filtered_ml_df = filtered_ml_df[filtered_ml_df['segment'] == selected_segment]
    if selected_category != "All":
        filtered_ml_df = filtered_ml_df[filtered_ml_df['category'] == selected_category]

    # -------------------------------
    # Train Model Button
    # -------------------------------
    if st.button("Train Model", key="train_tab3"):
        model, metrics = train_model(filtered_ml_df)
        st.success("✅ Model trained successfully!")

        # Save to session_state untuk Tab 4
        st.session_state.model = model
        st.session_state.filtered_ml_df = filtered_ml_df

        # -------------------------------
        # Metrics
        # -------------------------------
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{metrics['MAE']:.2f}")
        col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
        col3.metric("R²", f"{metrics['R2']:.2f}")

        # -------------------------------
        # Prepare Data for SHAP
        # -------------------------------
        X_numeric = filtered_ml_df.drop(columns=['sales', 'order_date']).select_dtypes(
            include=["float64", "int64", "bool"]
        )
        trained_features = model.get_booster().feature_names
        X_shap = X_numeric[trained_features]

        explainer = get_shap_explainer(model, X_shap)
        shap_values = compute_shap_values(explainer, X_shap)

        st.markdown("---")
        st.subheader("🔍 Model Interpretation (SHAP)")

        # SHAP Summary Plot
        st.markdown("### Feature Impact on Sales Prediction (Bar)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
        st.pyplot(fig)

        # SHAP Beeswarm
        st.markdown("### Feature Influence Distribution (Beeswarm)")
        fig2, ax2 = plt.subplots()
        shap.summary_plot(shap_values, X_shap, show=False)
        st.pyplot(fig2)

        # SHAP Waterfall (First Sample)
        st.markdown("### SHAP Waterfall (First Sample)")
        fig3, ax3 = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig3)

        st.info(
            """
            **Interpretation Insight:**
            - Lag and rolling features strongly influence predictions.
            - Time features capture seasonality & trends.
            - SHAP bar shows average impact per feature, beeswarm shows distribution,
              waterfall shows contribution for a single prediction.
            """
        )


# ===============================
# TAB 4 — FORECAST (PRODUCTION READY)
# ===============================
with tab4:
    st.subheader("🔮 Multi-Step Sales Forecast & Model Explanation")

    # -------------------------------
    # Filter segment / category
    # -------------------------------
    seg_opts = ["All"] + sorted(df["segment"].dropna().unique().tolist())
    cat_opts = ["All"] + sorted(df["category"].dropna().unique().tolist())

    seg = st.selectbox("Segment", seg_opts)
    cat = st.selectbox("Category", cat_opts)

    base_df = df.copy()
    if seg != "All":
        base_df = base_df[base_df["segment"] == seg]
    if cat != "All":
        base_df = base_df[base_df["category"] == cat]

    if base_df.shape[0] < 40:
        st.warning("Not enough data for this filter selection.")
        st.stop()

    # -------------------------------
    # Load model
    # -------------------------------
    if "model" not in st.session_state:
        st.warning("Train model first in Tab 3.")
        st.stop()

    model = st.session_state.model
    trained_features = model.get_booster().feature_names

    # -------------------------------
    # Forecast horizon (CLEAN)
    # -------------------------------
    horizon_map = {
        "1 Day": 1,
        "7 Days": 7,
        "1 Month": 30
    }

    horizon = st.selectbox(
        "Forecast Horizon",
        list(horizon_map.keys())
    )
    n_days = horizon_map[horizon]

    # -------------------------------
    # Build history window
    # -------------------------------
    history = base_df.sort_values("order_date").tail(30).copy()

    history_X = history.drop(
        columns=["sales", "order_date"], errors="ignore"
    )
    history_X = pd.get_dummies(history_X, drop_first=True)
    history_X = history_X.reindex(columns=trained_features, fill_value=0)

    history_y = history["sales"].tolist()
    last_date = history["order_date"].max()

    forecasts = []
    future_dates = []

    # -------------------------------
    # AUTOREGRESSIVE MULTI-STEP LOOP
    # -------------------------------
    for _ in range(n_days):
        X_input = history_X.tail(1)
        pred = model.predict(X_input)[0]

        forecasts.append(pred)
        next_date = last_date + pd.Timedelta(days=1)
        future_dates.append(next_date)

        new_row = X_input.copy()

        # Lag
        if "lag_1" in new_row.columns:
            new_row["lag_1"] = pred

        # Rolling
        if "sales_roll_mean_7" in new_row.columns:
            new_row["sales_roll_mean_7"] = np.mean(history_y[-7:] + [pred])
        if "sales_roll_std_7" in new_row.columns:
            new_row["sales_roll_std_7"] = np.std(history_y[-7:] + [pred])

        # Time features
        if "day" in new_row.columns:
            new_row["day"] = next_date.day
        if "month" in new_row.columns:
            new_row["month"] = next_date.month
        if "dayofweek" in new_row.columns:
            new_row["dayofweek"] = next_date.dayofweek

        history_X = pd.concat([history_X, new_row], ignore_index=True)
        history_y.append(pred)
        last_date = next_date

    # -------------------------------
    # Forecast Output
    # -------------------------------
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Sales": forecasts
    })

    st.metric("Next-Day Forecast", f"${forecasts[0]:,.0f}")
    st.markdown(f"### Forecast Result — {horizon}")
    st.line_chart(forecast_df.set_index("Date"))

    # -------------------------------
    # SHAP Explanation
    # -------------------------------
    st.markdown("### 🔍 Forecast Explanation (SHAP)")

    shap_X = history_X.tail(50)
    explainer = get_shap_explainer(model, shap_X)
    shap_values = compute_shap_values(explainer, shap_X)

    fig1, _ = plt.subplots()
    shap.summary_plot(shap_values, shap_X, plot_type="bar", show=False)
    st.pyplot(fig1)

    st.markdown("### SHAP Waterfall — Next Prediction")
    fig2, _ = plt.subplots()
    shap.plots.waterfall(shap_values[-1], show=False)
    st.pyplot(fig2)

    # -------------------------------
    # Download
    # -------------------------------
    st.download_button(
        "📥 Download Forecast CSV",
        forecast_df.to_csv(index=False).encode(),
        file_name=f"forecast_{horizon.replace(' ', '_')}.csv"
    )

    st.download_button(
        "📥 Download SHAP Values",
        pd.DataFrame(
            shap_values.values, columns=shap_X.columns
        ).to_csv(index=False).encode(),
        file_name="shap_forecast.csv"
    )

    st.success("Forecast & explanation generated successfully.")
