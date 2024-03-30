import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Set page config
st.set_page_config(page_title='GlucoGuardian', page_icon=':dna:')

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

# Inject custom CSS with the markdown
st.markdown(sidebar_style, unsafe_allow_html=True)

# dataset from GitHub
df = pd.read_csv("https://raw.githubusercontent.com/gopiashokan/dataset/main/diabetes_prediction_dataset.csv")

# Preprocessing using Ordinal Encoder
enc = OrdinalEncoder()
df["smoking_history"] = enc.fit_transform(df[["smoking_history"]])
df["gender"] = enc.fit_transform(df[["gender"]])

# Define Independent and Dependent Variables
x = df.drop("diabetes", axis=1)
y = df["diabetes"]

# 70% data - Train and 30% data - Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# RandomForest Algorithm
model = RandomForestClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

st.markdown(f'<h1 style="text-align: center; color: white;">Diabetes Prediction App</h1>', unsafe_allow_html=True)

# Sidebar for input fields
with st.sidebar:
    st.header('Patient Information')
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    age = st.slider('Age', min_value=0, max_value=120, value=30, step=1)
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
    smoking_history = st.selectbox('Smoking History', ['Never', 'Current', 'Former', 'Ever', 'Not Current', 'No Info'])
    bmi = st.slider('BMI', min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    hba1c_level = st.slider('HbA1c Level', min_value=0.0, max_value=15.0, value=5.7, step=0.1)
    blood_glucose_level = st.slider('Blood Glucose Level', min_value=0, max_value=300, value=100, step=1)

    submit = st.button('Submit')

# Prediction result display
if submit:
    gender_dict = {'Female': 0.0, 'Male': 1.0, 'Other': 2.0}
    hypertension_dict = {'No': 0, 'Yes': 1}
    heart_disease_dict = {'No': 0, 'Yes': 1}
    smoking_history_dict = {'Never': 4.0, 'No Info': 0.0, 'Current': 1.0, 'Former': 3.0, 'Ever': 2.0, 'Not Current': 5.0}

    try:
        user_data = np.array([[gender_dict[gender], age, hypertension_dict[hypertension], heart_disease_dict[heart_disease],
                               smoking_history_dict[smoking_history], bmi, hba1c_level, blood_glucose_level]])

        test_result = model.predict(user_data)

        if test_result[0] == 0:
            st.success('Diabetes Result: Negative')
            st.balloons()
        else:
            st.error('Diabetes Result: Positive (Please Consult with Doctor)')

    except Exception as e:
        st.warning(f'An error occurred: {e}')
        st.warning('Please fill all the required information correctly.')
