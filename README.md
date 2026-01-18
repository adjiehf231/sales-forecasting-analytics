
# 📈 Sales Forecasting & Analytics Dashboard

An **end-to-end data science project** that transforms raw sales transactions into an **interactive analytics and forecasting application** using  **Streamlit, XGBoost, and SHAP** .

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

The dataset undergoes a structured preprocessing pipeline:

### 1️⃣ Data Cleaning

* Convert `order_date` to datetime
* Remove invalid and missing records
* Normalize column formats

### 2️⃣ Time-Based Features

* `day`
* `month`
* `year`
* `dayofweek`

### 3️⃣ Lag Features

* `lag_1`: previous day's sales

  Enables autoregressive forecasting

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

This project emphasizes  **trustworthy AI** :

* **SHAP Summary (Bar):** Global feature importance
* **SHAP Beeswarm:** Feature influence distribution
* **SHAP Waterfall:** Contribution of each feature to a single prediction
* **SHAP Dependence:** Feature interaction insights (forecast tab)

---

## 🏗️ Project Structure

<pre class="overflow-visible! px-0!" data-start="3463" data-end="4004"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>sales</span><span>-forecasting-streamlit</span><span>/
│
├── app.py                     </span><span># Streamlit application</span><span>
├── </span><span>data</span><span>/
│   └── raw/
│       └── sales.csv          </span><span># Raw dataset</span><span>
│
├── src/
│   ├── data_loader.py         </span><span># Data loading logic</span><span>
│   ├── preprocessing.py       </span><span># Cleaning & formatting</span><span>
│   ├── features.py            </span><span># Feature engineering</span><span>
│   ├── train.py               </span><span># Model training & evaluation</span><span>
│   └── explain.py             </span><span># SHAP explainability</span><span>
│
├── model.pkl                  </span><span># Trained model (optional)</span><span>
├── requirements.txt
└── README.md
</span></span></code></div></div></pre>

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

<pre class="overflow-visible! px-0!" data-start="4295" data-end="4639"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span># Clone repository</span><span>
git </span><span>clone</span><span> https://github.com/your-username/sales-forecasting-streamlit.git
</span><span>cd</span><span> sales-forecasting-streamlit

</span><span># Create virtual environment</span><span>
python -m venv .venv
.\.venv\Scripts\Activate.ps1

</span><span># Install dependencies</span><span>
pip install -r requirements.txt

</span><span># Run application</span><span>
streamlit run app.py
</span></span></code></div></div></pre


---
## 🔗 Project Links


* **🌐 Live Demo (Streamlit App):**

  [https://adjieh231-sales-forecasting-analytics.streamlit.app/](https://adjieh231-sales-forecasting-analytics.streamlit.app/)
* **📂 GitHub Repository:**

  [https://github.com/adjiehf231/sales-forecasting-analytics](https://github.com/adjiehf231/sales-forecasting-analytics)
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

📌 Python • Machine Learning • Forecasting • ASAP
