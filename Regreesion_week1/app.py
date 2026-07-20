import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# Helper: embed a local image as an inline <img> tag (for use inside HTML strings)
# ============================================================
def money_icon_tag(size=22):
    with open("money_icon.png", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{encoded}" width="{size}" style="vertical-align:middle; margin-right:4px;">'

MONEY_ICON = money_icon_tag()
MONEY_ICON_SM = money_icon_tag(16)

# ============================================================
# Page config + fonts + custom styling
# ============================================================
st.set_page_config(page_title="Ames House Price Predictor", page_icon="house_icon.png", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0;
        background: linear-gradient(90deg, #1b998b, #f46036);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #9a9a9a;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }
    .price-box {
        background: linear-gradient(135deg, #1b998b 0%, #0e5c53 100%);
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 28px rgba(27,153,139,0.28);
    }
    .price-box h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        margin: 0.2rem 0 0 0;
        color: white;
    }
    .price-box p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    .kpi-card {
        background: #f7f7f9;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #ececec;
    }
    .kpi-card h3 {
        font-family: 'Space Grotesk', sans-serif;
        margin: 0;
        font-size: 1.35rem;
        color: #1b998b;
    }
    .kpi-card p {
        margin: 0;
        font-size: 0.8rem;
        color: #999;
    }
    .tier-card {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #ececec;
        text-align: center;
    }
    .tier-label {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        padding: 0.6rem 0;
        font-size: 1.05rem;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load data + train models (cached)
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")

@st.cache_resource
def train_models(df):
    features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]
    X = df[features]
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression().fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

    metrics = {}
    for name, m in [("Linear Regression", lr_model), ("Random Forest", rf_model)]:
        pred = m.predict(X_test)
        metrics[name] = {
            "R2": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        }

    return lr_model, rf_model, features, metrics

df = load_data()
lr_model, rf_model, features, metrics = train_models(df)

# ============================================================
# Header
# ============================================================
h_col1, h_col2 = st.columns([0.06, 0.94])
with h_col1:
    st.image("house_icon.png", width=55)
with h_col2:
    st.markdown('<p class="main-header">Ames House Price Predictor</p>', unsafe_allow_html=True)

st.markdown('<p class="sub-header">Estimate a home\'s sale price and see how it compares '
            'to 1,460 real house sales in Ames, Iowa.</p>', unsafe_allow_html=True)

# --- Dataset KPI strip (money-themed icons) ---
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><h3>🏠 {len(df):,}</h3><p>Homes in Dataset</p></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><h3>{MONEY_ICON}${df["SalePrice"].median():,.0f}</h3><p>Median Price</p></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><h3>📊 ${df["SalePrice"].min():,.0f}–${df["SalePrice"].max():,.0f}</h3><p>Price Range</p></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><h3>🏗️ {df["YearBuilt"].min():.0f}–{df["YearBuilt"].max():.0f}</h3><p>Years Built</p></div>', unsafe_allow_html=True)

st.write("")

# ============================================================
# Layout: inputs left, results right
# ============================================================
input_col, result_col = st.columns([1, 1.3], gap="large")

with input_col:
    st.subheader("🔧 House Features")

    model_choice = st.segmented_control(
        "Model", ["Linear Regression", "Random Forest"], default="Random Forest"
    ) if hasattr(st, "segmented_control") else st.radio(
        "Model", ["Linear Regression", "Random Forest"], horizontal=True
    )

    c1, c2 = st.columns(2)
    with c1:
        overall_qual = st.slider("Overall Quality", 1, 10, 6, help="1 = Very Poor, 10 = Very Excellent")
        garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
        year_built = st.slider("Year Built", 1870, 2010, 2000)
    with c2:
        gr_liv_area = st.number_input("Living Area (sq ft)", 500, 6000, 1500, step=50)
        total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 6000, 1000, step=50)

    st.divider()
    with st.expander("💹 About the models (accuracy metrics)"):
        m = metrics[model_choice]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("📈 R² Score", f"{m['R2']:.3f}")
        mc2.markdown(f'<p style="font-size:0.8rem; color:#888; margin-bottom:0;">{MONEY_ICON_SM}MAE</p>'
                     f'<p style="font-size:1.4rem; font-weight:600; margin-top:0;">${m["MAE"]:,.0f}</p>',
                     unsafe_allow_html=True)
        mc3.metric("📉 RMSE", f"${m['RMSE']:,.0f}")
        st.caption("Metrics calculated on a held-out 20% test set from the Ames Housing dataset.")

# --- Prediction ---
input_data = pd.DataFrame([[overall_qual, gr_liv_area, garage_cars, total_bsmt_sf, year_built]],
                           columns=features)
model = lr_model if model_choice == "Linear Regression" else rf_model
prediction = model.predict(input_data)[0]
percentile = (df["SalePrice"] < prediction).mean() * 100

with result_col:
    st.subheader("💰 Estimated Price")
    st.markdown(f"""
    <div class="price-box">
        <p>{MONEY_ICON}Predicted Sale Price ({model_choice})</p>
        <h1>${prediction:,.0f}</h1>
        <p>📈 Higher than {percentile:.0f}% of homes in the dataset</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.caption(f"Dataset median: ${df['SalePrice'].median():,.0f}  •  "
               f"Dataset average: ${df['SalePrice'].mean():,.0f}")

# ============================================================
# Price tier photo section (uses your own local images)
# ============================================================
st.divider()
st.subheader("🏘️ What Tier Does This House Fall Into?")

tiers = [
    {"name": "Budget Home", "range": "< $150K", "max": 150000, "file": "images/budget_home.jpg", "icon": "🏚️"},
    {"name": "Mid-Range Home", "range": "$150K – $300K", "max": 300000, "file": "images/midrange_home.jpg", "icon": "🏡"},
    {"name": "Luxury Home", "range": "> $300K", "max": float("inf"), "file": "images/luxury_home.jpg", "icon": "🏰"},
]

if prediction < 150000:
    active_tier = "Budget Home"
elif prediction < 300000:
    active_tier = "Mid-Range Home"
else:
    active_tier = "Luxury Home"

tcols = st.columns(3)
for tcol, tier in zip(tcols, tiers):
    with tcol:
        is_active = tier["name"] == active_tier
        border_color = "#1b998b" if is_active else "#ececec"
        st.markdown(f'<div class="tier-card" style="border: 3px solid {border_color};">',
                    unsafe_allow_html=True)
        if os.path.exists(tier["file"]):
            st.image(tier["file"], use_container_width=True)
        else:
            st.markdown(
                f'<div style="height:160px; display:flex; align-items:center; '
                f'justify-content:center; font-size:3rem; background:#f2f2f2;">{tier["icon"]}</div>',
                unsafe_allow_html=True
            )
        label_bg = "#1b998b" if is_active else "#f7f7f9"
        label_color = "white" if is_active else "#333"
        star = " ⭐" if is_active else ""
        st.markdown(
            f'<div class="tier-label" style="background:{label_bg}; color:{label_color};">'
            f'{tier["icon"]} {tier["name"]}{star}</div>',
            unsafe_allow_html=True
        )
        st.caption(f"Typical range: {tier['range']}")
        st.markdown('</div>', unsafe_allow_html=True)

if not any(os.path.exists(t["file"]) for t in tiers):
    st.info(
        "📸 Add your own photos to show up here: create an `images/` folder next to `app.py` "
        "with `budget_home.jpg`, `midrange_home.jpg`, and `luxury_home.jpg`. "
        "Free, legal real US house photos: search 'house exterior' on unsplash.com or pexels.com "
        "and download — no attribution required."
    )

# ============================================================
# Tabs for visualizations
# ============================================================
st.divider()
st.subheader("📊 Explore the Data")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price vs Living Area", "📦 Price by Quality", "🌡️ Correlations",
    "📉 Price Distribution", "⭐ Feature Importance", "🏗️ Price by Year Built"
])

with tab1:
    fig1 = px.scatter(df, x="GrLivArea", y="SalePrice", opacity=0.35,
                       color_discrete_sequence=["#1b998b"],
                       labels={"GrLivArea": "Living Area (sq ft)", "SalePrice": "Sale Price ($)"},
                       title="Where your house lands among all homes")
    fig1.add_trace(go.Scatter(x=[gr_liv_area], y=[prediction], mode="markers",
                               marker=dict(size=18, color="#f46036", symbol="star",
                                           line=dict(width=1, color="black")),
                               name="Your House"))
    fig1.update_layout(height=440, font_family="Inter")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    qual_colors = px.colors.sequential.Teal
    fig2 = px.box(df, x="OverallQual", y="SalePrice", color="OverallQual",
                  color_discrete_sequence=qual_colors,
                  labels={"OverallQual": "Overall Quality", "SalePrice": "Sale Price ($)"},
                  title="Price distribution at each quality level")
    fig2.add_trace(go.Scatter(x=[overall_qual], y=[prediction], mode="markers",
                               marker=dict(size=18, color="#f46036", symbol="star",
                                           line=dict(width=1, color="black")),
                               name="Your House"))
    fig2.update_layout(height=440, showlegend=False, font_family="Inter")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    corr_cols = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt"]
    corr = df[corr_cols].corr()
    fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                      title="How each feature relates to price and to each other")
    fig3.update_layout(height=440, font_family="Inter")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    fig4 = px.histogram(df, x="SalePrice", nbins=40, color_discrete_sequence=["#1b998b"],
                         labels={"SalePrice": "Sale Price ($)"},
                         title="Distribution of sale prices across all homes")
    fig4.add_vline(x=prediction, line_width=3, line_dash="dash", line_color="#f46036",
                    annotation_text="Your House 💰", annotation_position="top")
    fig4.add_vline(x=df["SalePrice"].median(), line_width=2, line_dash="dot", line_color="gray",
                    annotation_text="Median", annotation_position="bottom")
    fig4.update_layout(height=440, font_family="Inter")
    st.plotly_chart(fig4, use_container_width=True)

with tab5:
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig5 = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Sunset",
                  title="What drives the Random Forest model's predictions")
    fig5.update_layout(height=440, showlegend=False, font_family="Inter")
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Higher importance = the model relies on this feature more heavily when predicting price.")

with tab6:
    year_avg = df.groupby("YearBuilt")["SalePrice"].mean().reset_index()
    fig6 = px.line(year_avg, x="YearBuilt", y="SalePrice", markers=True,
                    color_discrete_sequence=["#1b998b"],
                    labels={"YearBuilt": "Year Built", "SalePrice": "Average Sale Price ($)"},
                    title="Average sale price by construction year")
    fig6.add_vline(x=year_built, line_width=2, line_dash="dash", line_color="#f46036",
                    annotation_text="Your Year", annotation_position="top")
    fig6.update_layout(height=440, font_family="Inter")
    st.plotly_chart(fig6, use_container_width=True)

st.divider()
st.caption("Built with Streamlit, scikit-learn, and Plotly · Ames Housing dataset (1,460 homes)")
