import streamlit as st
import pandas as pd
import joblib

#Load model
model = joblib.load("./notebook/CustomerChurn_RFModel.pkl")

st.set_page_config(
     page_title= 'Customer Churn Prediction'
)

st.sidebar.title("About")
st.sidebar.info(
    '''
    A Customer Churn Prediction App which lets you know a customer likely to churn or not
    
    Model is trained using Random Forest Classifier
    
    Features:
    - Age
    - Gender
    - Tenure
    - Monthly Charges
    - Total Charges
    - Contract Type
    - Internet Service
    - Tech Support
    '''
)

st.title("Customer Churn Prediction")
st.divider()
st.markdown("### Provide the following details: ###")


# User input
age = st.slider("Age", 18, 100, 30)
gender = st.selectbox(
  "Sex",
  ["Male", "Female"]
)
tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=150,
    value=12
)
monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)
total_charges = st.number_input(
       "Total Charges",
       min_value = 0.0,
       max_value= 20000.0,
       value = 2000.0
)
contract_type = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "None"]
)
tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)



# prediction button

if st.button("Predict Churn"):
    input_df = pd.DataFrame({
        'Age': [age],
        'Tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Gender_Male': [1 if gender == 'male' else 0],
        'ContractType_One-Year': [1 if contract_type == 'One year' else 0],
        'ContractType_Two-Year': [1 if contract_type == 'Two year' else 0],
        'InternetService_Fiber Optic': [1 if internet_service == 'Fiber optics' else 0],
        'InternetService_Unknown': [1 if internet_service == 'None' else 0],
        'TechSupport_Yes': [1 if tech_support == 'Yes' else 0]
    })
  
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
            st.error("⚠️ Customer is likely to churn.")
    else:
            st.success("✅ Customer is unlikely to churn.")

    st.divider()

    st.subheader("Prediction Probability")

    st.progress(float(probability))

    col1, col2 = st.columns([2,0.5])

    with col1:
           st.metric(
                  "Churn Probability",
                  f'{probability: .2%}'
           )
    with col2:
           st.metric(
                  "Retention Probability",
                  f'{1-probability: .2%}'
           )