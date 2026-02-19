# Simple UI for our Salary Prediction model
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# App config & title
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Prediction Portal")
st.write("Enter your years of experience to see the predicted salary based on our model.")

df = pd.read_csv('salaries.csv')

@st.cache_data
def train_model(df):
    df = df.dropna()
    X = df[['years_of_experience']]
    y = df['salary']
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    
    return model, mae

model, mae = train_model(df)

# User input
user_exp = st.number_input("Years of Experience:", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# Prediction Logic
test_df = pd.DataFrame({'years_of_experience': [user_exp]})
predicted_salary = model.predict(test_df)[0]

# Return / Display results
st.markdown(f"### Predicted Salary: **${predicted_salary:,.2f}**")

st.divider()

# Print model params
st.subheader("Model Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Intercept", f"{model.intercept_:.2f}")
with col2:
    st.metric("Coefficient", f"{model.coef_[0]:.2f}")
with col3:
    st.metric("MAE", f"${mae:,.2f}")