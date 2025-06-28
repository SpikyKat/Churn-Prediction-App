# 🔁 Dynamic Customer Churn Prediction App

A powerful, interactive Streamlit dashboard to predict customer churn using machine learning, explain predictions with SHAP values, and generate LLM-powered insights using LLaMA via Groq API.

---

## 🚀 Features

- 📥 Upload your own labeled dataset to train models
- 📤 Upload new data for churn prediction
- 🧠 LLM-powered churn reasons using LLaMA3 (via Groq API)
- 📊 Compare model performance: 
  - XGBoost
  - Random Forest
  - Logistic Regression
- 🔍 Filter results by feature (e.g. Geography, Contract Type)
- 🎯 Visual KPIs (Total customers, churn %, avg probability)
- 📉 Confusion Matrix & ROC Curve
- 🧬 SHAP-based Explainability:
  - Waterfall Plot
  - Summary Bar Plot
  - Beeswarm Plot
  - Text Summary of Top Features
- 📥 Download prediction results

---

## 🗂️ Folder Structure

project-root/
│
├── data/ # CSV files (training + prediction)

├── notebooks/ # Jupyter notebooks (optional analysis)

├── streamlit_app/ # Main app script (app.py)

├── .env # Groq API Key (DO NOT COMMIT!)

├── .gitignore # Git ignored files

├── requirements.txt # Dependencies

└── README.md # This file

## 🧪 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
```
### 2️⃣Create .env File
Create a file named .env in the root directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the App
```bash
streamlit run streamlit_app/app.py
```
### 🔐 Environment Variables
Variable	Description
GROQ_API_KEY	Your API key from Groq

### 📦 Requirements
All dependencies are listed in requirements.txt.
Major packages used:

- streamlit

- pandas, scikit-learn, xgboost

- matplotlib, seaborn

- shap

- groq (for LLaMA3-powered insights)

### 🧠 LLM-Powered Insights
Using LLaMA 3 via the Groq API, the app analyzes SHAP feature importance and generates plain-language answers to:

"Why are customers likely churning in this dataset?"

These insights are shown in a dedicated section at the bottom of the dashboard.

### 📸 Screenshots 

![image](https://github.com/user-attachments/assets/31bcbb03-c80a-440a-82e5-5e3d5e2e816d)
![image](https://github.com/user-attachments/assets/3cd41f19-4285-4c0b-a90e-d54f1d399991)
![image](https://github.com/user-attachments/assets/05e9ac11-238d-45fc-924c-8e06dd5ffe6d)
![image](https://github.com/user-attachments/assets/21f54062-6e50-46e4-bcb1-561ac11f220f)
![image](https://github.com/user-attachments/assets/bdb9a9c8-0f5c-4983-b9e5-4fe9c73026f1)

### 📄 License
MIT License - feel free to use, modify, and share.

### ✨ Credits
Built with ❤️ using:

- Streamlit

- SHAP

- Groq API

- scikit-learn

- XGBoost
