# 📈 Sales Forecasting & Analytics Dashboard

An **end-to-end data science project** that transforms raw sales transactions into an **interactive analytics and forecasting application** using **Streamlit, XGBoost, and SHAP**.

This project demonstrates **data preprocessing, exploratory analysis, machine learning, forecasting, and model interpretability** in a single production-ready dashboard.

---

## 🚀 Live Application Features

### 📊 Dashboard

* High-level sales KPIs:
  * Total Revenue
  * Average Sales
  * Maximum Daily Sales
  * Average Year-over-Year Growth
* Interactive daily sales trend
* Date range filtering for dynamic analysis

### 🔍 Exploratory Data Analysis (EDA)

* Sales distribution analysis
* Correlation analysis between engineered features
* Transparent preprocessing pipeline explanation
* Preview of final dataset used for modeling

### 🤖 Machine Learning

* XGBoost regression model for sales prediction
* Performance evaluation using:
  * MAE
  * RMSE
  * R² Score
* Model interpretation with SHAP:
  * Feature importance (bar)
  * Feature impact distribution (beeswarm)
  * Local explanation (waterfall)

### 🔮 Forecasting

* Multi-step autoregressive forecasting:
  * 1 Day
  * 7 Days
  * 1 Month
* Segment & category-based forecasting
* Dynamic feature updates (lag, rolling statistics, time features)
* SHAP explanation for forecasted values
* Downloadable forecast & SHAP reports

---

## 🧠 Business Questions Answered

* How do sales evolve over time?
* What patterns and seasonality exist in historical sales?
* Which features drive sales predictions the most?
* How will sales likely behave in the near future?
* How do segment and category affect future sales?

---

## 🗂️ Dataset

* **Source:** Transactional retail sales data
* **Granularity:** Daily sales
* **Key Columns:**
  * `order_date`
  * `sales`
  * `segment`
  * `category`
  * Product and customer attributes

---

## 🛠️ Data Preprocessing & Feature Engineering

### 1️⃣ Data Cleaning

* Convert `order_date` to datetime
* Remove invalid and missing records
* Normalize column formats

### 2️⃣ Time-Based Features

* `day`, `month`, `year`, `dayofweek`

### 3️⃣ Lag Features

* `lag_1`: previous day's sales (for autoregressive forecasting)

### 4️⃣ Rolling Statistics

* `sales_roll_mean_7`: short-term trend
* `sales_roll_std_7`: short-term volatility

### 5️⃣ Final Dataset

* Numeric features only
* Consistent structure across:
  * EDA
  * Model training
  * Forecasting

---

## 🤖 Machine Learning Model

* **Algorithm:** XGBoost Regressor
* **Why XGBoost?**
  * Handles non-linear patterns
  * Robust to skewed distributions
  * Strong performance on tabular data

### Model Evaluation Metrics

* **MAE:** Mean Absolute Error
* **RMSE:** Root Mean Squared Error
* **R²:** Coefficient of Determination

---

## 🔍 Model Explainability (SHAP)

* **SHAP Summary (Bar):** Global feature importance
* **SHAP Beeswarm:** Feature influence distribution
* **SHAP Waterfall:** Contribution of each feature to a single prediction
* **SHAP Dependence:** Feature interaction insights (forecast tab)

---

## 🏗️ Project Structure

sales-forecasting-streamlit/
│
├── app.py # Streamlit application
├── data/
│ └── raw/
│ └── sales.csv # Raw dataset
├── src/
│ ├── data_loader.py # Data loading logic
│ ├── preprocessing.py # Cleaning & formatting
│ ├── features.py # Feature engineering
│ ├── train.py # Model training & evaluation
│ └── explain.py # SHAP explainability
├── model.pkl # Trained model (optional)
├── requirements.txt
└── README.md

---

## ⚙️ Tech Stack

* **Python**
* **Streamlit** – interactive dashboard
* **Pandas & NumPy** – data processing
* **Plotly & Matplotlib** – visualization
* **XGBoost** – machine learning
* **SHAP** – model explainability
* **Joblib** – model persistence

---

## ▶️ How to Run Locally

```bash
# Clone repository
git clone https://github.com/adjiehf231/sales-forecasting-analytics.git
cd sales-forecasting-analytics

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
source .venv/bin/activate     # Linux / Mac

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---



## 🔗 Project Links

* **🌐 Live Demo:**

  https://adjiehf231-sales-forecasting-analytics.streamlit.app/
* **📂 GitHub Repository:**

  https://github.com/adjiehf231/sales-forecasting-analytics

---

## 📌 Key Takeaways

* Demonstrates **end-to-end data science workflow**
* Combines analytics, ML, forecasting, and explainability
* Production-ready Streamlit architecture
* Suitable for **Data Analyst / Data Scientist / ML Engineer portfolios**

---

## 👤 Author

**Adjie Hari Fajar**

Data Scientist | Data Analyst

📌 Python • Machine Learning • Forecasting • Streamlit

---
