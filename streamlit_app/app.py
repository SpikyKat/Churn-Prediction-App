import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from groq import Groq
from dotenv import load_dotenv
import os, uuid

# Load env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("🔁 Dynamic Churn Prediction App")

# Uploads
st.sidebar.subheader("📥 Upload Files")
train_file = st.sidebar.file_uploader("1. Upload Labeled Training Dataset (with 'Churn')", type=['csv'])
predict_file = st.sidebar.file_uploader("2. Upload Unlabeled Prediction Data", type=['csv'])

# Encoding
def encode_data(df, encoders=None):
    df = df.copy()
    encs = encoders or {}
    for col in df.select_dtypes(include='object').columns:
        if col not in encs:
            encs[col] = LabelEncoder()
            df[col] = encs[col].fit_transform(df[col])
        else:
            df[col] = encs[col].transform(df[col])
    return df, encs

# Model training
def train_models(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    models = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42)
    }
    results = {}
    for name, model in models.items():
        X_input = X_scaled if name == 'LogisticRegression' else X
        model.fit(X_input, y)
        y_pred = model.predict(X_input)
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, model.predict_proba(X_input)[:, 1])
        }
    return results

# LLM Insight using LLaMA
def generate_llm_insight(df: pd.DataFrame):
    df_sample = df.sample(min(100, len(df)))  # limit rows to keep prompt light
    prompt = f"""Analyze the following customer churn dataset and explain the top reasons why customers are likely to churn. Be concise and data-driven.\n\n{df_sample.head(20).to_markdown(index=False)}"""
    try:
        chat = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a senior data analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Failed to generate insights: {e}"

# Main logic
if train_file:
    df_train = pd.read_csv(train_file)
    if 'Churn' not in df_train.columns:
        st.error("Training data must include a 'Churn' column.")
    else:
        df_train['TotalCharges'] = pd.to_numeric(df_train.get('TotalCharges', pd.Series(dtype=float)), errors='coerce')
        df_train.dropna(inplace=True)
        y = df_train['Churn'].map({'Yes': 1, 'No': 0}) if df_train['Churn'].dtype == 'object' else df_train['Churn']
        X = df_train.drop(columns=['Churn', 'customerID'], errors='ignore')
        X_encoded, encoders = encode_data(X)

        st.subheader("📊 Training Data Preview")
        st.dataframe(df_train.head())

        st.subheader("📈 Model Training")
        results = train_models(X_encoded, y)
        model_df = pd.DataFrame(results).T[['accuracy', 'precision', 'recall', 'f1', 'auc']]
        st.dataframe(model_df.style.background_gradient(cmap="Blues"))
        best_model_name = model_df['auc'].idxmax()
        best_model = results[best_model_name]['model']
        st.success(f"✅ Best Model Selected: {best_model_name}")

        if predict_file:
            df_pred = pd.read_csv(predict_file)
            df_pred['TotalCharges'] = pd.to_numeric(df_pred.get('TotalCharges', pd.Series(dtype=float)), errors='coerce')
            df_pred.dropna(inplace=True)
            df_pred = df_pred.drop(columns=['customerID'], errors='ignore')
            df_pred = df_pred[[col for col in X.columns if col in df_pred.columns]]
            X_pred, _ = encode_data(df_pred, encoders)

            preds = best_model.predict(X_pred)
            probs = best_model.predict_proba(X_pred)[:, 1]
            df_result = df_pred.copy()
            df_result['Churn Prediction'] = ['Yes' if p == 1 else 'No' for p in preds]
            df_result['Churn Probability (%)'] = (probs * 100).round(2)

            st.subheader("📤 Predictions")
            st.dataframe(df_result)
            st.download_button("📥 Download Predictions", df_result.to_csv(index=False), "predictions.csv")

            # Filters
            st.sidebar.subheader("🔎 Filter Results")
            filter_col = st.sidebar.selectbox("Filter by Column", df_result.columns)
            val = st.sidebar.selectbox("Value", df_result[filter_col].unique())
            filtered = df_result[df_result[filter_col] == val]
            st.subheader(f"📌 Filtered View: {filter_col} = {val}")
            st.write(filtered)
            st.metric("Total Customers", len(filtered))
            st.metric("Predicted Churn Rate", f"{(filtered['Churn Prediction']=='Yes').mean()*100:.2f}%")
            st.metric("Avg Churn Probability", f"{filtered['Churn Probability (%)'].mean():.2f}%")

            # Evaluation
            st.subheader("📉 Confusion Matrix")
            y_train_pred = best_model.predict(X_encoded)
            cm = confusion_matrix(y, y_train_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            st.subheader("📈 ROC Curve")
            fpr, tpr, _ = roc_curve(y, best_model.predict_proba(X_encoded)[:, 1])
            fig2 = plt.figure(figsize=(4, 3))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, best_model.predict_proba(X_encoded)[:, 1]):.2f}")
            plt.plot([0, 1], [0, 1], '--')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC Curve")
            plt.legend()
            st.pyplot(fig2)

            # SHAP Explainability
            st.subheader("🔍 SHAP Waterfall Plot (Row 0)")
            explainer = shap.Explainer(best_model)
            shap_values = explainer(X_pred)
            shap.initjs()
            try:
                row_0 = shap_values[0] if len(shap_values[0].shape) == 1 else shap_values[0, 0]
                fig_waterfall = shap.plots.waterfall(row_0, max_display=10, show=False)
                st.pyplot(fig_waterfall)
            except Exception as e:
                st.warning(f"Could not display waterfall plot: {e}")

            st.subheader("🧾 Top 5 Feature Impacts (Row 0)")
            try:
                if shap_values.values.ndim == 3:
                    shap_row_values = shap_values.values[0, :, 1]
                else:
                    shap_row_values = shap_values.values[0]
                top_features = pd.Series(shap_row_values, index=X_pred.columns).sort_values(key=abs, ascending=False).head(5)
                st.dataframe(top_features.rename("Impact on Churn").to_frame())
            except Exception as e:
                st.warning(f"Could not compute SHAP text summary: {e}")

            st.subheader("📊 SHAP Summary Plot")
            try:
                fig_bar = plt.figure(figsize=(8, 6) if st.checkbox("Expand SHAP Summary") else (4, 3))
                shap_summary_vals = shap_values[..., 1] if shap_values.values.ndim == 3 else shap_values
                shap.plots.bar(shap_summary_vals, max_display=15, show=False)
                st.pyplot(fig_bar)
            except Exception as e:
                st.warning(f"Could not display SHAP summary plot: {e}")

            if st.checkbox("🐝 Show SHAP Beeswarm"):
                st.subheader("🐝 SHAP Beeswarm Plot")
                fig_beeswarm = plt.figure(figsize=(10, 6))
                shap.plots.beeswarm(shap_summary_vals, max_display=15, show=False)
                st.pyplot(fig_beeswarm)

            # 🧠 LLM Insight using LLaMA 3
            st.subheader("🧠 LLM Insight: Why Are Customers Churning?")
            if st.button("🪄 Generate Insight with LLaMA"):
                with st.spinner("Thinking..."):
                    insight = generate_llm_insight(df_train)
                    st.markdown(insight)
