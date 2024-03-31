import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Set page config
st.set_page_config(page_title='Diabetes Prediction App', page_icon=':dna:')

# Custom CSS to inject into the Streamlit page with a background image and a color gradient
sidebar_style = """
<style>
[data-testid="stSidebar"] > div:first-child {
    background-image: linear-gradient(rgba(75, 10, 118, 0.6), rgba(107, 9, 55, 0.6)), url('https://cdn.create.vista.com/api/media/small/629002736/stock-photo-blurred-african-american-businesswoman-diabetes-holding-insulin-syringe-office');
    background-size: cover;
    color: white;
}
[data-testid="stSidebar"] .css-1lcbmhc {
    background-color: rgba(255, 255, 255, 0);
}
</style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# Load your dataset
df = pd.read_csv("https://raw.githubusercontent.com/gopiashokan/dataset/main/diabetes_prediction_dataset.csv")

# Preprocess the data
enc = OrdinalEncoder()
df["smoking_history"] = enc.fit_transform(df[["smoking_history"]])
df["gender"] = enc.fit_transform(df[["gender"]])

# Define Independent and Dependent Variables
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Sidebar for patient input fields
with st.sidebar:
    st.header('Patient Information')
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    age = st.slider('Age', 0, 120, 30)
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
    smoking_history = st.selectbox('Smoking History', ['Never', 'Current', 'Former', 'Ever', 'Not Current', 'No Info'])
    bmi = st.slider('BMI', 10.0, 50.0, 22.0)
    hba1c_level = st.slider('HbA1c Level', 0.0, 15.0, 5.7)
    blood_glucose_level = st.slider('Blood Glucose Level', 0, 300, 100)
    submit = st.button('Submit')

# Main page description
st.write("# Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on health parameters.
Enter the required information in the sidebar and press 'Submit' to see the prediction.
""")

# Radar chart visualization
# Placeholder patient data and healthy ranges
df_metrics = pd.DataFrame({
    'Metric': ['Age', 'BMI', 'HbA1c Level', 'Blood Glucose Level'],
    'Value': [age, bmi, hba1c_level, blood_glucose_level],  # Use actual patient values
    'Healthy_min': [0, 18.5, 4.0, 70],  # These are hypothetical healthy minimums
    'Healthy_max': [120, 24.9, 5.7, 140]  # These are hypothetical healthy maximums
})

# Normalize the values
df_metrics['Normalized_Value'] = df_metrics.apply(
    lambda row: (row['Value'] - row['Healthy_min']) / (row['Healthy_max'] - row['Healthy_min']), axis=1)

# Create radar chart
fig = px.line_polar(df_metrics, r='Normalized_Value', theta='Metric', line_close=True)
fig.update_traces(fill='toself')
st.plotly_chart(fig)

# Prediction result display
if submit:
    # Perform prediction with the model and display results
    # Note: Replace 'enc' with the actual OrdinalEncoder and input values with actual patient inputs
    prediction = model.predict(np.array([[0, age, 0 if hypertension == 'No' else 1, 0 if heart_disease == 'No' else 1, 4.0, bmi, hba1c_level, blood_glucose_level]]))
    st.write("## Patient Results")
    if prediction[0] == 0:
        st.success('The model predicts: No diabetes')
    else:
        st.error('The model predicts: Diabetes')
    
    # Generate and display the feature importance graph
    feature_names = X.columns
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'features': feature_names, 'importance': feature_importances})
    importance_df = importance_df.sort_values('importance', ascending=False)

    fig = px.bar(importance_df, x='features', y='importance', title='Feature Importance')
    fig.update_layout(xaxis_title='Feature', yaxis_title='Importance Score')
    st.write(fig)

    # Automatic summary of feature importances
    st.write("### Summary")
    st.write(f"The most important features in predicting diabetes are {importance_df.iloc[0, 0]} and {importance_df.iloc[1, 0]}, indicating they have the strongest influence on the model's predictions.")

    #Install plotly to run radar plots
