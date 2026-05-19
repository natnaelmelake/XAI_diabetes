import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report
)

# ─────────────────────────────────────────────────────────────
# PATHS — adjust if your folder names are different
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # app/
ROOT_DIR   = os.path.dirname(BASE_DIR)                    # diabetes_xai/
DATA_PATH  = os.path.join(ROOT_DIR, "data",   "diabetes_binary_health_indicators_BRFSS2015.csv")
MODEL_DIR  = os.path.join(ROOT_DIR, "models")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes XAI Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F0F6FA; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #065A82;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .highlight-card {
        background: #EBF5FB;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #AED6F1;
        margin-bottom: 1rem;
    }
    .badge-blue {
        background: #065A82; color: white;
        padding: 3px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600;
    }
    .badge-green {
        background: #02C39A; color: white;
        padding: 3px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600;
    }
    .badge-warn {
        background: #E07B39; color: white;
        padding: 3px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600;
    }
    div[data-testid="stSidebar"] { background-color: #021B3A; }
    div[data-testid="stSidebar"] p { color: #9DCDE8; }
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 { color: white; }
    div[data-testid="stSidebar"] .stSelectbox label { color: #9DCDE8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FEATURE LABELS — human readable names
# ─────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "HighBP":               "High Blood Pressure",
    "HighChol":             "High Cholesterol",
    "CholCheck":            "Cholesterol Check (last 5 yrs)",
    "BMI":                  "Body Mass Index (BMI)",
    "Smoker":               "Smoker (100+ cigarettes lifetime)",
    "Stroke":               "Ever had a Stroke",
    "HeartDiseaseorAttack": "Heart Disease or Attack",
    "PhysActivity":         "Physical Activity (past 30 days)",
    "Fruits":               "Eats Fruit Daily",
    "Veggies":              "Eats Vegetables Daily",
    "HvyAlcoholConsump":    "Heavy Alcohol Consumption",
    "AnyHealthcare":        "Has Healthcare Coverage",
    "NoDocbcCost":          "Could Not See Doctor (cost)",
    "GenHlth":              "General Health (1=Excellent–5=Poor)",
    "MentHlth":             "Poor Mental Health Days (0–30)",
    "PhysHlth":             "Poor Physical Health Days (0–30)",
    "DiffWalk":             "Difficulty Walking",
    "Sex":                  "Sex (0=Female, 1=Male)",
    "Age":                  "Age Group (1=18–24 to 13=80+)",
    "Education":            "Education Level (1–6)",
    "Income":               "Income Level (1–8)",
}

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean["BMI"] = df_clean["BMI"].clip(upper=60)
    int_cols = [c for c in df_clean.columns if c != "BMI"]
    df_clean[int_cols] = df_clean[int_cols].astype(int)
    return df, df_clean

# ─────────────────────────────────────────────────────────────
# LOAD MODELS FROM PICKLE
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained models...")
def load_models(df_clean):
    # ── load pickle files ──
    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
        xgb = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "lr_model.pkl"), "rb") as f:
        lr = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "shap_values.pkl"), "rb") as f:
        shap_data = pickle.load(f)

    # ── recreate train/test split (same seed = same split) ──
    X = df_clean.drop("Diabetes_binary", axis=1)
    y = df_clean["Diabetes_binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── SHAP explainer (fast for XGBoost) ──
    shap_explainer = shap.TreeExplainer(xgb)

    # ── LIME explainer ──
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["No Diabetes", "Diabetes"],
        mode="classification",
        random_state=42
    )

    # ── compute metrics on test set ──
    def get_metrics(model, Xt, yt):
        yp  = model.predict(Xt)
        ypr = model.predict_proba(Xt)[:, 1]
        rep = classification_report(
            yt, yp,
            target_names=["No Diabetes", "Diabetes"],
            output_dict=True
        )
        return {
            "Accuracy":           round(accuracy_score(yt, yp), 3),
            "ROC-AUC":            round(roc_auc_score(yt, ypr), 3),
            "Diabetes Recall":    round(rep["Diabetes"]["recall"], 2),
            "Diabetes Precision": round(rep["Diabetes"]["precision"], 2),
            "Diabetes F1":        round(rep["Diabetes"]["f1-score"], 2),
        }

    perf = {
        "Logistic Regression": get_metrics(lr,  X_test, y_test),
        "Random Forest":       get_metrics(rf,  X_test, y_test),
        "XGBoost":             get_metrics(xgb, X_test, y_test),
    }

    return {
        "lr":              lr,
        "rf":              rf,
        "xgb":             xgb,
        "X_train":         X_train,
        "X_test":          X_test,
        "y_train":         y_train,
        "y_test":          y_test,
        "sample_1000":     shap_data["sample"],
        "shap_values":     shap_data["shap_values"],
        "shap_explainer":  shap_explainer,
        "lime_explainer":  lime_explainer,
        "perf":            perf,
        "feature_names":   X_train.columns.tolist(),
    }

# ─────────────────────────────────────────────────────────────
# HELPER — matplotlib figure style
# ─────────────────────────────────────────────────────────────
def style_ax(ax, fig):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#F0F6FA")
    fig.patch.set_facecolor("#F0F6FA")

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Diabetes XAI")
    st.markdown("---")
    page = st.selectbox("Navigate", [
        "🏠  Home",
        "📊  EDA Dashboard",
        "🤖  Model Performance",
        "🔍  Predict & Explain",
        "⚖️   SHAP vs LIME Consistency",
    ])
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("CDC BRFSS 2015")
    st.markdown("**Models**")
    st.markdown("LR · RF · XGBoost · ANN")
    st.markdown("**XAI Methods**")
    st.markdown("SHAP · LIME")
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#9DCDE8'>"
        "Final Year Project · 2024–2025</p>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────
# LOAD DATA (runs on every page)
# ─────────────────────────────────────────────────────────────
df_raw, df_clean = load_data()

# ═════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown("# 🩺 Explainable ML for Diabetes Prediction")
    st.markdown(
        "### Evaluating the Consistency of SHAP and LIME "
        "on the CDC BRFSS 2015 Dataset"
    )
    st.markdown("---")

    # top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients",  "253,680")
    c2.metric("After Cleaning",  "229,474")
    c3.metric("Features",        "21")
    c4.metric("Diabetic Cases",  "13.9%")

    st.markdown("---")
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### 🎯 Research Objectives")
        for n, title, desc in [
            ("1", "Compare 4 ML models",
             "Logistic Regression, Random Forest, XGBoost, ANN "
             "using 5-fold stratified CV"),
            ("2", "Apply SHAP & LIME",
             "Generate global and local explanations for the "
             "best performing model"),
            ("3", "Evaluate consistency",
             "Formally measure agreement between SHAP and LIME "
             "at global and local levels"),
            ("4", "Build this app",
             "Present predictions and explanations in a "
             "clinician-friendly interface"),
        ]:
            st.markdown(f"""
            <div class='metric-card'>
                <span class='badge-blue'>{n}</span>
                <strong style='color:#021B3A;margin-left:8px'>{title}</strong><br>
                <span style='color:#64748B;font-size:0.9rem'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("### 📋 Research Questions")
        for rq, q, col in [
            ("RQ1",
             "Which ML model achieves best predictive performance "
             "on the CDC BRFSS 2015 dataset?",
             "#065A82"),
            ("RQ2",
             "What are the most important features identified "
             "by SHAP and LIME?",
             "#1C7293"),
            ("RQ3",
             "To what extent do SHAP and LIME produce consistent "
             "explanations for the same model predictions?",
             "#028090"),
        ]:
            st.markdown(f"""
            <div style='background:white;border-radius:10px;padding:1rem;
                        border-left:4px solid {col};margin-bottom:0.8rem;
                        box-shadow:0 2px 6px rgba(0,0,0,0.06)'>
                <span style='font-weight:700;color:{col}'>{rq}</span>
                <p style='margin:0.3rem 0 0;color:#021B3A;
                          font-size:0.92rem'>{q}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔑 Key Finding")
        st.markdown("""
        <div class='highlight-card'>
            <strong>80%</strong> local consistency &nbsp;·&nbsp;
            <strong>70%</strong> global consistency &nbsp;·&nbsp;
            <strong>100%</strong> directional agreement
            between SHAP and LIME
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗂️ Dataset Overview")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Feature Types**")
        st.dataframe(pd.DataFrame({
            "Type":    ["Binary (0/1)", "Ordinal scale", "Continuous"],
            "Count":   [14, 6, 1],
            "Examples":["HighBP, Smoker, Stroke",
                        "GenHlth, Age, Income", "BMI"],
        }), use_container_width=True, hide_index=True)
    with d2:
        st.markdown("**Preprocessing Steps**")
        st.dataframe(pd.DataFrame({
            "Step":  ["Remove duplicates", "Cap BMI outliers",
                      "Type conversion",   "Train-test split"],
            "Detail":["24,206 rows removed",
                      "805 rows capped at BMI = 60",
                      "Binary/ordinal → int",
                      "80/20 stratified (random_state=42)"],
        }), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ═════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Class Distribution",
        "BMI Analysis",
        "Feature Comparison",
        "Correlation Heatmap",
    ])

    # ── Tab 1 ──
    with tab1:
        st.markdown("### Class Distribution — Target Variable")
        counts = df_clean["Diabetes_binary"].value_counts()
        pcts   = df_clean["Diabetes_binary"].value_counts(normalize=True) * 100

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.metric("No Diabetes (0)", f"{counts[0]:,}",
                      f"{pcts[0]:.1f}%")
            st.metric("Diabetes (1)",    f"{counts[1]:,}",
                      f"{pcts[1]:.1f}%")
            st.markdown("""
            <div class='highlight-card'>
            <strong>⚠️ Class Imbalance</strong><br>
            86 / 14 split. A naive model predicting "No Diabetes"
            every time would score 86% accuracy without learning
            anything. We use ROC-AUC and Diabetic Recall instead.
            </div>
            """, unsafe_allow_html=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(["No Diabetes", "Diabetes"],
                          [counts[0], counts[1]],
                          color=["#065A82", "#E07B39"],
                          edgecolor="white", width=0.5)
            for bar, cnt in zip(bars, [counts[0], counts[1]]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 800,
                        f"{cnt:,}", ha="center",
                        fontsize=11, fontweight="bold")
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title("Diabetes Class Distribution",
                         fontsize=13, fontweight="bold")
            style_ax(ax, fig)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Tab 2 ──
    with tab2:
        st.markdown("### BMI Distribution by Diabetes Status")
        diabetic     = df_clean[df_clean["Diabetes_binary"] == 1]["BMI"]
        non_diabetic = df_clean[df_clean["Diabetes_binary"] == 0]["BMI"]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(non_diabetic, bins=40, alpha=0.6,
                color="#065A82", label="No Diabetes", density=True)
        ax.hist(diabetic,     bins=40, alpha=0.6,
                color="#E07B39", label="Diabetes",    density=True)
        ax.set_xlabel("BMI", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("BMI Distribution by Diabetes Status",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        style_ax(ax, fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**BMI Stats — No Diabetes**")
            st.dataframe(
                non_diabetic.describe().round(2).to_frame("BMI"),
                use_container_width=True
            )
        with col2:
            st.markdown("**BMI Stats — Diabetes**")
            st.dataframe(
                diabetic.describe().round(2).to_frame("BMI"),
                use_container_width=True
            )

        st.markdown("""
        <div class='highlight-card'>
        <strong>Key observation:</strong> Diabetic patients show a right-shifted
        BMI distribution peaking around 29–31 compared to 26–27 for non-diabetic
        patients, consistent with the +0.217 correlation between BMI and diabetes.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3 ──
    with tab3:
        st.markdown("### Binary Feature Comparison by Diabetes Status")
        binary_features = [
            "HighBP", "HighChol", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity",
            "Fruits", "Veggies", "HvyAlcoholConsump", "DiffWalk"
        ]
        d1 = df_clean[df_clean["Diabetes_binary"]==1][binary_features].mean()*100
        d0 = df_clean[df_clean["Diabetes_binary"]==0][binary_features].mean()*100

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(binary_features))
        w = 0.35
        ax.bar(x - w/2, d0.values, w, label="No Diabetes",
               color="#065A82", alpha=0.85)
        ax.bar(x + w/2, d1.values, w, label="Diabetes",
               color="#E07B39", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(binary_features, rotation=40,
                           ha="right", fontsize=9)
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.set_title("Binary Feature Comparison by Diabetes Status (%)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        style_ax(ax, fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class='highlight-card'>
        <strong>Key patterns:</strong> HighBP (75% vs 40%), HighChol (67% vs 40%),
        and DiffWalk (37% vs 15%) are all notably higher in diabetic patients.
        PhysActivity is lower in diabetic patients (63% vs 75%) confirming
        exercise as a protective factor.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4 ──
    with tab4:
        st.markdown("### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(14, 10))
        corr = df_clean.corr()
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4,
            annot_kws={"size": 7}, ax=ax, square=True
        )
        ax.set_title("Feature Correlation Heatmap",
                     fontsize=14, fontweight="bold")
        fig.patch.set_facecolor("#F0F6FA")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class='highlight-card'>
        <strong>Notable pairs:</strong>
        GenHlth & PhysHlth (0.52) &nbsp;·&nbsp;
        GenHlth & DiffWalk (0.45) &nbsp;·&nbsp;
        PhysHlth & DiffWalk (0.47) &nbsp;·&nbsp;
        Education & Income (0.42).
        No pair exceeds 0.6 — no severe multicollinearity.
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown("# 🤖 Model Performance Comparison")
    st.markdown("---")

    models_data = load_models(df_clean)
    perf        = models_data["perf"]

    # CV scores from our training runs
    cv_scores = {
        "Logistic Regression": {"CV ROC-AUC": 0.808, "CV Std": 0.001},
        "Random Forest":       {"CV ROC-AUC": 0.812, "CV Std": 0.001},
        "XGBoost":             {"CV ROC-AUC": 0.814, "CV Std": 0.001},
        "ANN":                 {"CV ROC-AUC": 0.814, "CV Std": 0.001},
    }

    # ANN test results (hardcoded — trained in Colab)
    ann_perf = {
        "Accuracy":           0.696,
        "ROC-AUC":            0.819,
        "Diabetes Recall":    0.81,
        "Diabetes Precision": 0.31,
        "Diabetes F1":        0.45,
    }

    # ── CV summary cards ──
    st.markdown("### 📈 Cross-Validation Results (5-Fold Stratified)")
    cv_cols = st.columns(4)
    for i, (name, cv) in enumerate(cv_scores.items()):
        with cv_cols[i]:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-weight:700;color:#021B3A;
                            margin-bottom:4px'>{name}</div>
                <div style='font-size:1.8rem;font-weight:800;
                            color:#065A82'>{cv["CV ROC-AUC"]}</div>
                <div style='font-size:0.8rem;color:#64748B'>
                    ROC-AUC &nbsp;±&nbsp; {cv["CV Std"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Full Test Set Comparison")

    all_perf = {**perf, "ANN": ann_perf}
    rows = []
    for model_name, m in all_perf.items():
        rows.append({
            "Model":               model_name,
            "CV ROC-AUC":         cv_scores.get(model_name, {}).get("CV ROC-AUC", "—"),
            "Test ROC-AUC":       m["ROC-AUC"],
            "Accuracy":           m["Accuracy"],
            "Diabetes Recall":    m["Diabetes Recall"],
            "Diabetes Precision": m["Diabetes Precision"],
            "Diabetes F1":        m["Diabetes F1"],
        })
    perf_df = pd.DataFrame(rows)

    def highlight_best(col):
        if col.name in ["CV ROC-AUC", "Test ROC-AUC",
                        "Diabetes Recall", "Diabetes F1"]:
            best = col.max()
            return [
                "background-color:#D6EAF8;font-weight:bold"
                if v == best else ""
                for v in col
            ]
        return [""] * len(col)

    st.dataframe(
        perf_df.style.apply(highlight_best),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    st.markdown("### 📊 Visual Comparison")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        names  = perf_df["Model"].tolist()
        cv_auc = perf_df["CV ROC-AUC"].tolist()
        te_auc = perf_df["Test ROC-AUC"].tolist()
        x = np.arange(len(names))
        ax.bar(x - 0.2, cv_auc, 0.35, label="CV ROC-AUC",
               color="#065A82", alpha=0.85)
        ax.bar(x + 0.2, te_auc, 0.35, label="Test ROC-AUC",
               color="#1C7293", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0.78, 0.84)
        ax.set_ylabel("ROC-AUC", fontsize=11)
        ax.set_title("CV vs Test ROC-AUC",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        style_ax(ax, fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        recall  = perf_df["Diabetes Recall"].tolist()
        colors  = ["#E07B39" if r == max(recall) else "#065A82"
                   for r in recall]
        bars = ax.bar(names, recall, color=colors, alpha=0.85)
        for bar, val in zip(bars, recall):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    str(val), ha="center",
                    fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Diabetic Recall", fontsize=11)
        ax.set_title("Diabetic Recall by Model\n(most critical metric)",
                     fontsize=12, fontweight="bold")
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        style_ax(ax, fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""
    <div class='highlight-card'>
    <strong>💡 Key Insight — Why accuracy is misleading:</strong><br>
    Initial Random Forest achieved <strong>84.4% accuracy</strong> but only
    <strong>15% diabetic recall</strong> — missing 85% of actual diabetic patients.
    After tuning with <code>balanced_subsample</code> and depth constraints,
    recall improved to <strong>73%</strong> at the cost of lower overall accuracy (74%).
    This illustrates why accuracy is not the right metric for imbalanced medical datasets.
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT & EXPLAIN
# ═════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown("# 🔍 Predict & Explain")
    st.markdown(
        "Enter patient details below to get a diabetes risk prediction "
        "with SHAP and LIME explanations side by side."
    )
    st.markdown("---")

    models_data   = load_models(df_clean)
    xgb_model     = models_data["xgb"]
    lime_explainer = models_data["lime_explainer"]
    shap_explainer = models_data["shap_explainer"]
    feat_names     = models_data["feature_names"]

    # ── Input form ──
    st.markdown("### 🧑‍⚕️ Patient Features")
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Clinical Indicators**")
            HighBP       = st.selectbox(
                "High Blood Pressure", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
            HighChol     = st.selectbox(
                "High Cholesterol", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
            CholCheck    = st.selectbox(
                "Cholesterol Check (last 5 yrs)", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
            BMI          = st.slider("BMI", 12, 60, 28)
            Stroke       = st.selectbox(
                "Ever had a Stroke", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
            HeartDisease = st.selectbox(
                "Heart Disease or Attack", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
            DiffWalk     = st.selectbox(
                "Difficulty Walking", [0, 1],
                format_func=lambda x: "Yes" if x else "No")

        with col2:
            st.markdown("**Lifestyle**")
            Smoker   = st.selectbox(
                "Smoker (100+ cigarettes lifetime)", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
            PhysAct  = st.selectbox(
                "Physical Activity (past 30 days)", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
            Fruits   = st.selectbox(
                "Eats Fruit Daily", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
            Veggies  = st.selectbox(
                "Eats Vegetables Daily", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
            HvyAlc   = st.selectbox(
                "Heavy Alcohol Consumption", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
            MentHlth = st.slider(
                "Poor Mental Health Days (0–30)", 0, 30, 0)
            PhysHlth = st.slider(
                "Poor Physical Health Days (0–30)", 0, 30, 0)

        with col3:
            st.markdown("**Demographics & Access**")
            GenHlth   = st.slider(
                "General Health (1=Excellent, 5=Poor)", 1, 5, 3)
            Age       = st.slider(
                "Age Group (1=18–24 to 13=80+)", 1, 13, 7)
            Sex       = st.selectbox(
                "Sex", [0, 1],
                format_func=lambda x: "Male" if x else "Female")
            Education = st.slider(
                "Education Level (1=None, 6=College)", 1, 6, 4)
            Income    = st.slider(
                "Income Level (1=<$10k, 8=$75k+)", 1, 8, 4)
            AnyHC     = st.selectbox(
                "Has Healthcare Coverage", [0, 1],
                format_func=lambda x: "Yes" if x else "No")
            NoDoc     = st.selectbox(
                "Could Not See Doctor (cost)", [0, 1],
                format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button(
            "🔍 Predict & Explain", use_container_width=True
        )

    if submitted:
        # assemble patient array in correct column order
        patient = np.array([[
            HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
            HeartDisease, PhysAct, Fruits, Veggies, HvyAlc,
            AnyHC, NoDoc, GenHlth, MentHlth, PhysHlth,
            DiffWalk, Sex, Age, Education, Income
        ]])
        patient_df = pd.DataFrame(patient, columns=feat_names)

        prob = xgb_model.predict_proba(patient)[0][1]
        pred = int(prob >= 0.5)

        st.markdown("---")
        st.markdown("### 📋 Prediction Result")

        r1, r2, r3 = st.columns(3)
        with r1:
            color = "#E07B39" if pred == 1 else "#02C39A"
            label = "⚠️ Diabetic Risk" if pred == 1 else "✅ Low Risk"
            st.markdown(f"""
            <div style='background:{color};border-radius:12px;
                        padding:1.5rem;text-align:center;color:white'>
                <div style='font-size:1.2rem;font-weight:700'>{label}</div>
                <div style='font-size:2.5rem;font-weight:800'>
                    {prob*100:.1f}%
                </div>
                <div style='font-size:0.9rem'>Diabetes Probability</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            log_odds = np.log(prob / (1 - prob + 1e-9))
            st.metric("Prediction Threshold", "50%")
            st.metric("Log-odds  f(x)", f"{log_odds:.3f}")
        with r3:
            st.markdown("""
            <div class='highlight-card'>
            <strong>How to interpret:</strong><br>
            Probability above 50% → predicted diabetic.<br>
            The explanations below show <em>why</em> the model
            made this prediction for this specific patient.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        exp1, exp2 = st.columns(2)

        # ── SHAP explanation ──
        with exp1:
            st.markdown("### 🔵 SHAP Explanation")
            shap_vals   = shap_explainer.shap_values(patient_df)
            feat_imp    = dict(zip(feat_names, shap_vals[0]))
            sorted_feat = sorted(
                feat_imp.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]

            names_s  = [FEATURE_LABELS.get(f, f) for f, _ in sorted_feat]
            vals_s   = [v for _, v in sorted_feat]
            colors_s = ["#E07B39" if v > 0 else "#065A82" for v in vals_s]

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(names_s[::-1], vals_s[::-1],
                    color=colors_s[::-1], alpha=0.85)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=10)
            ax.set_title(
                "SHAP Feature Attribution\n"
                "(orange = increases risk · blue = decreases risk)",
                fontsize=11, fontweight="bold"
            )
            style_ax(ax, fig)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("**Top 3 SHAP Drivers:**")
            for feat, val in sorted_feat[:3]:
                direction = "🔴 increases" if val > 0 else "🔵 decreases"
                label = FEATURE_LABELS.get(feat, feat)
                st.markdown(
                    f"- **{label}** → {direction} risk "
                    f"({val:+.3f})"
                )

        # ── LIME explanation ──
        with exp2:
            st.markdown("### 🟠 LIME Explanation")
            with st.spinner("Generating LIME explanation..."):
                lime_result = lime_explainer.explain_instance(
                    data_row=patient[0],
                    predict_fn=xgb_model.predict_proba,
                    num_features=10
                )
            lime_list   = lime_result.as_list()
            lime_feats  = [f for f, _ in lime_list]
            lime_vals   = [v for _, v in lime_list]
            colors_l    = [
                "#E07B39" if v > 0 else "#065A82"
                for v in lime_vals
            ]

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(lime_feats[::-1], lime_vals[::-1],
                    color=colors_l[::-1], alpha=0.85)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("LIME Weight (local contribution)", fontsize=10)
            ax.set_title(
                "LIME Feature Attribution\n"
                "(orange = towards diabetes · blue = away from diabetes)",
                fontsize=11, fontweight="bold"
            )
            style_ax(ax, fig)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("**Top 3 LIME Drivers:**")
            top3_lime = sorted(
                lime_list, key=lambda x: abs(x[1]), reverse=True
            )[:3]
            for feat, val in top3_lime:
                direction = "🔴 increases" if val > 0 else "🔵 decreases"
                st.markdown(
                    f"- **{feat}** → {direction} risk "
                    f"({val:+.3f})"
                )

        # ── Quick consistency check ──
        st.markdown("---")
        st.markdown("### ⚖️ Quick SHAP vs LIME Consistency")

        shap_top5 = set([
            f for f, _ in sorted(
                feat_imp.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
        ])
        lime_top5 = set()
        for feat_str, _ in lime_list[:5]:
            for fn in feat_names:
                if fn in feat_str:
                    lime_top5.add(fn)
                    break

        overlap = shap_top5 & lime_top5
        score   = len(overlap) / 5 * 100

        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            st.markdown("**SHAP Top 5**")
            for f in shap_top5:
                st.markdown(f"- {FEATURE_LABELS.get(f, f)}")
        with qc2:
            st.markdown("**LIME Top 5**")
            for f in lime_top5:
                st.markdown(f"- {FEATURE_LABELS.get(f, f)}")
        with qc3:
            color = "#02C39A" if score >= 60 else "#E07B39"
            st.markdown(f"""
            <div style='background:{color};border-radius:10px;
                        padding:1rem;text-align:center;color:white'>
                <div style='font-size:2rem;font-weight:800'>
                    {score:.0f}%
                </div>
                <div style='font-weight:600'>Consistency Score</div>
                <div style='font-size:0.85rem'>
                    {len(overlap)} of 5 features agree
                </div>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 5 — SHAP vs LIME CONSISTENCY
# ═════════════════════════════════════════════════════════════
elif "Consistency" in page:
    st.markdown("# ⚖️ SHAP vs LIME Consistency Analysis")
    st.markdown(
        "This page presents the formal consistency evaluation — "
        "the primary research contribution of this project."
    )
    st.markdown("---")

    # ── Summary score cards ──
    st.markdown("### 🏆 Consistency Summary")
    sc1, sc2, sc3 = st.columns(3)
    for col_obj, val, label, sub, color in [
        (sc1, "80%", "Local Consistency",
         "4 of 5 top features agreed\nfor individual patients", "#065A82"),
        (sc2, "70%", "Global Consistency",
         "7 of 10 top features agreed\nacross full dataset", "#1C7293"),
        (sc3, "100%", "Directional Agreement",
         "Zero directional contradictions\nbetween SHAP and LIME", "#028090"),
    ]:
        with col_obj:
            st.markdown(f"""
            <div style='background:{color};border-radius:12px;
                        padding:1.5rem;text-align:center;color:white;
                        margin-bottom:1rem'>
                <div style='font-size:3rem;font-weight:800'>{val}</div>
                <div style='font-weight:700'>{label}</div>
                <div style='font-size:0.85rem;margin-top:4px'>
                    {sub}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Global importance data ──
    shap_global = {
        "GenHlth": 0.57, "HighBP": 0.52, "BMI": 0.40,
        "Age": 0.39, "HighChol": 0.29, "Sex": 0.12,
        "Income": 0.10, "MentHlth": 0.09,
        "CholCheck": 0.08, "HvyAlcoholConsump": 0.07,
    }
    lime_global = {
        "GenHlth": 0.091, "HighBP": 0.089, "Stroke": 0.081,
        "HeartDiseaseorAttack": 0.074, "Age": 0.074,
        "HighChol": 0.071, "BMI": 0.054,
        "HvyAlcoholConsump": 0.052, "NoDocbcCost": 0.030,
        "Income": 0.021,
    }

    # ── Side-by-side global charts ──
    st.markdown("### 📊 Global Feature Importance: SHAP vs LIME")
    gc1, gc2 = st.columns(2)

    with gc1:
        fig, ax = plt.subplots(figsize=(6, 5))
        features = list(shap_global.keys())
        values   = list(shap_global.values())
        colors   = [
            "#E07B39" if f in lime_global else "#065A82"
            for f in features
        ]
        ax.barh(features[::-1], values[::-1],
                color=colors[::-1], alpha=0.85)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
        ax.set_title(
            "SHAP Global Importance\n"
            "(orange = also in LIME top 10)",
            fontsize=11, fontweight="bold"
        )
        style_ax(ax, fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with gc2:
        fig, ax = plt.subplots(figsize=(6, 5))
        features_l = list(lime_global.keys())
        values_l   = list(lime_global.values())
        colors_l   = [
            "#E07B39" if f in shap_global else "#1C7293"
            for f in features_l
        ]
        ax.barh(features_l[::-1], values_l[::-1],
                color=colors_l[::-1], alpha=0.85)
        ax.set_xlabel("Mean |LIME Weight|", fontsize=10)
        ax.set_title(
            "LIME Global Importance\n"
            "(orange = also in SHAP top 10)",
            fontsize=11, fontweight="bold"
        )
        style_ax(ax, fig)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Local consistency table ──
    st.markdown("### 📋 Local Consistency — Top 5 Features per Patient")
    st.dataframe(pd.DataFrame({
        "Rank":                      [1, 2, 3, 4, 5],
        "SHAP Global Top 5":         ["GenHlth","HighBP","BMI",
                                      "Age","HighChol"],
        "LIME Top 5 (Diabetic)":     ["Age","Stroke","HighBP",
                                      "HighChol","GenHlth"],
        "LIME Top 5 (Non-Diabetic)": ["Age","HighBP",
                                      "HeartDiseaseorAttack",
                                      "HighChol","GenHlth"],
        "Match (Diabetic)":          ["✅","❌","✅","✅","✅"],
        "Match (Non-Diabetic)":      ["✅","✅","❌","✅","✅"],
    }), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Global consistency table ──
    st.markdown("### 📋 Global Consistency — Top 10 Features")
    st.dataframe(pd.DataFrame({
        "Rank":        list(range(1, 11)),
        "SHAP Feature":list(shap_global.keys()),
        "SHAP Value":  list(shap_global.values()),
        "LIME Feature":list(lime_global.keys()),
        "LIME Value":  list(lime_global.values()),
        "Match":       ["✅","✅","❌","❌","✅",
                        "✅","✅","✅","❌","✅"],
    }), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Interpretation ──
    st.markdown("### 💡 Interpretation")
    st.markdown("""
    <div class='metric-card'>
        <strong>Why do they agree?</strong><br>
        Both methods consistently identify the same clinically
        validated diabetes risk factors — General Health, High
        Blood Pressure, Age and High Cholesterol — regardless of
        their different mathematical foundations. This suggests
        these features carry genuine predictive signal that any
        explanation method will detect.
    </div>
    <div class='metric-card'>
        <strong>Why do they diverge?</strong><br>
        SHAP computes exact Shapley values globally across all
        feature value combinations. LIME fits a local linear
        approximation in the neighbourhood of each prediction.
        LIME elevates Stroke and HeartDiseaseorAttack because
        binary conditions have strong local effects. SHAP ranks
        BMI higher because it captures its continuous non-linear
        contribution globally. These are complementary
        perspectives, not contradictions.
    </div>
    <div class='highlight-card'>
        <strong>✅ Clinical Implication:</strong>
        The 100% directional agreement means both methods agree on
        <em>whether</em> each feature increases or decreases
        diabetes risk — the most important question for clinical
        decision support. Using both methods together provides a
        more complete picture than either alone.
    </div>
    """, unsafe_allow_html=True)
