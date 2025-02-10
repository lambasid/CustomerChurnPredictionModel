import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import utils as ut
from openai import OpenAI
import scipy.stats as stats
import requests

# Initialize OpenAI client with environment variable for API key
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ['GROQ_API_KEY']
)

# --- Streamlit App Setup ---

# Display the application title
st.title("Customer Churn Prediction")

# --- Data Collection Section ---
# Collect customer data through user input fields
customer_data = {
    "CreditScore": st.number_input("Credit Score", min_value=300, max_value=850, value=600),
    "Geography": st.selectbox("Geography", ["France", "Germany", "Spain"]),
    "Gender": st.radio("Gender", ["Male", "Female"]),
    "Age": st.number_input("Age", min_value=18, max_value=100, value=40),
    "Tenure": st.number_input("Tenure", min_value=0, max_value=10, value=3),
    "Balance": st.number_input("Balance", min_value=0.0, value=60000.0),
    "NumOfProducts": st.number_input("Number of Products", min_value=1, max_value=4, value=2),
    "HasCrCard": st.checkbox("Has Credit Card", value=True, key="has_credit_card"),  # Ensures a unique key for the checkbox
    "IsActiveMember": st.checkbox("Is Active Member", value=True, key="is_active_member"),  # Unique key for checkbox
    "EstimatedSalary": st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
}

# --- Model Prediction Trigger ---
# Send a POST request to an external FastAPI server when the 'Predict Churn' button is clicked
if st.button("Predict Churn"):
    response = requests.post("https://churnpredictionmodel.onrender.com/predict", json=customer_data)
    if response.status_code == 200:
        st.write("Churn Prediction:", response.json())
    else:
        st.write("Error:", response.status_code, response.text)

# --- Data Preparation Functions ---
def prepare_input_opt(input_df):
    """Prepare input DataFrame by adding derived features and ensuring correct column order."""
    # Calculate derived features
    input_df['CLV'] = input_df['Balance'] * input_df['EstimatedSalary'] / 100000  # Customer Lifetime Value
    input_df['TenureAgeRatio'] = input_df['Tenure'] / input_df['Age']             # Tenure-to-Age Ratio

    # Create age group features
    input_df['AgeGroup_MiddleAge'] = np.where((input_df['Age'] >= 40) & (input_df['Age'] < 60), 1, 0)
    input_df['AgeGroup_Senior'] = np.where(input_df['Age'] >= 60, 1, 0)
    input_df['AgeGroup_Elderly'] = np.where(input_df['Age'] >= 75, 1, 0)

    # Define expected feature columns and reorder DataFrame accordingly
    expected_columns = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary', 'Geography_France', 'Geography_Germany',
        'Geography_Spain', 'Gender_Female', 'Gender_Male', 'CLV', 'TenureAgeRatio',
        'AgeGroup_MiddleAge', 'AgeGroup_Senior', 'AgeGroup_Elderly'
    ]
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)  # Ensure all expected columns exist
    return input_df

def calculate_percentiles(selected_customer, df):
    """Calculate percentiles for various metrics based on customer data."""
    percentiles = {
        'CreditScore Percentile': stats.percentileofscore(df['CreditScore'], selected_customer['CreditScore']),
        'Age Percentile': stats.percentileofscore(df['Age'], selected_customer['Age']),
        'Tenure Percentile': stats.percentileofscore(df['Tenure'], selected_customer['Tenure']),
        'Balance Percentile': stats.percentileofscore(df['Balance'], selected_customer['Balance']),
        'NumOfProducts Percentile': stats.percentileofscore(df['NumOfProducts'], selected_customer['NumOfProducts']),
        'EstimatedSalary Percentile': stats.percentileofscore(df['EstimatedSalary'], selected_customer['EstimatedSalary']),
    }
    return percentiles

# --- Model Explanation and Utility Functions ---
def explain_prediction(probability, input_dict, surname):
    """Generate a detailed explanation of the churn prediction based on customer data."""
    prompt = f"""You are an expert data scientist..."""  # Truncated for brevity
    print("EXPLAINATION PROMPT: ", prompt)
    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}]
    )
    return raw_response.choices[0].message.content

def load_model(filename):
    """Load a machine learning model from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
    """Prepare input data for prediction by creating a feature dictionary and DataFrame."""
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

def make_predictions(input_df, input_dict):
    """Make predictions using multiple models and display results."""
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbour': knn_model.predict_proba(input_df)[0][1],
    }
    avg_probability = np.mean(list(probabilities.values()))
    
    # Display prediction results in Streamlit columns
    col1, col2 = st.columns(2)
    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {avg_probability:.2%} probability of churning.")
    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)
    
    return avg_probability

def generate_email(probability, input_dict, explanation, surname):
    """Generate a personalized email to the customer based on prediction results."""
    prompt = f"""You are a manager..."""  # Truncated for brevity
    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}]
    )
    print("\n\nEMAIL PROMPT: ", prompt)
    return raw_response.choices[0].message.content

# --- Model Initialization ---
xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')

# --- Main Customer Data Handling ---
df = pd.read_csv("churn.csv")  # Load customer data from CSV
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

# Allow user to select a customer
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    # Extract customer ID and surname from selection
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_surname = selected_customer_option.split(" - ")[1]

    # Retrieve selected customer data
    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

    # Display input fields for customer data
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=int(selected_customer['CreditScore']))
        location = st.selectbox("Location", ["Spain", "France", "Germany"], index=["Spain", "France", "Germany"].index(selected_customer['Geography']))
        gender = st.radio("Gender", ["Male", "Female"], index=0 if selected_customer['Gender'] == 'Male' else 1)
        age = st.number_input("Age", min_value=0, max_value=100, value=int(selected_customer['Age']))
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=50, value=int(selected_customer['Tenure']))
    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=float(selected_customer['Balance']))
        num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=int(selected_customer['NumOfProducts']))
        has_credit_card = st.checkbox("Has Credit Card", value=bool(selected_customer['HasCrCard']))
        is_active_member = st.checkbox("Is Active Member", value=bool(selected_customer['IsActiveMember']))
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=float(selected_customer['EstimatedSalary']))

    # Prepare input data and make predictions
    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)
    avg_probability = make_predictions(input_df, input_dict)

    # Generate and display prediction explanation
    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)

    # Generate and display personalized email
    email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])
    st.subheader("Personalized Email")
    st.markdown(email)
