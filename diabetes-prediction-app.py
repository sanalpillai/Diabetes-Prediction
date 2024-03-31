import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from streamlit_extras.stoggle import stoggle
from streamlit_extras.let_it_rain import rain 


# Set page config
st.set_page_config(page_title='Diabetes Prediction App', page_icon=':dna:')

# Custom CSS for the sidebar
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

# Load and preprocess the dataset
df = pd.read_csv("https://raw.githubusercontent.com/sanalpillai/Diabetes-Prediction-Capstone/main/Dataset/diabetes_prediction_dataset.csv?token=GHSAT0AAAAAACKT2DNC32NG3XF3AAI4AUEAZQJRRJA")
enc = OrdinalEncoder()
df[["smoking_history"]] = enc.fit_transform(df[["smoking_history"]])
df[["gender"]] = enc.fit_transform(df[["gender"]])

# Define Independent and Dependent Variables
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Define dictionaries for encoding categorical variables
gender_dict = {'Male': 0, 'Female': 1, 'Other': 2}
hypertension_dict = {'No': 0, 'Yes': 1}
heart_disease_dict = {'No': 0, 'Yes': 1}
smoking_history_dict = {'Never': 0, 'Current': 1, 'Former': 2, 'Ever': 3, 'Not Current': 4, 'No Info': 5}

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

# Define the path to your images
image_paths = {
    'Male': 'https://github.com/sanalpillai/Diabetes-Prediction-Capstone/blob/main/Assets/Male.png',
    'Female': 'https://github.com/sanalpillai/Diabetes-Prediction-Capstone/blob/main/Assets/Female.png',
    'Other': 'https://github.com/sanalpillai/Diabetes-Prediction-Capstone/blob/main/Assets/Others.png'
}

# Radar chart visualization
df_metrics = pd.DataFrame({
    'Metric': ['Age', 'BMI', 'HbA1c', 'Blood Glucose'],
    'Value': [age, bmi, hba1c_level, blood_glucose_level],  # User inputs
    'Healthy_min': [0, 18.5, 4.0, 70],  # example healthy minimums
    'Healthy_max': [120, 24.9, 5.7, 140]  # example healthy maximums
})

# Normalize the values for the radar chart
df_metrics['Normalized_Value'] = df_metrics.apply(
    lambda row: (row['Value'] - row['Healthy_min']) / (row['Healthy_max'] - row['Healthy_min']), axis=1)

# Create radar chart
fig = px.line_polar(df_metrics, r='Normalized_Value', theta='Metric', line_close=True)
fig.update_traces(fill='toself')
st.plotly_chart(fig)

# Prediction result display
if submit:
    st.write("## Patient Results")
    # Display patient results and profile photo side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Display the patient's profile photo
        st.image(image_paths[gender], width=150, caption=f"Patient Profile: {gender}")
    
    with col2:
        # Display patient information with ideal ranges or values for each factor
        st.markdown(f"""
        - Gender: {gender} (Ideal: Not a direct risk factor)
        - Age: {age} years old (Ideal range: <45 years old, as age can be a risk factor)
        - Hypertension: {hypertension} (Ideal: No, as hypertension can increase risk)
        - Heart Disease: {heart_disease} (Ideal: No, as heart disease can increase risk)
        - Smoking History: {smoking_history} (Ideal: Never, as smoking can increase risk)
        - BMI: {bmi} (Ideal range: 18.5-24.9, as higher values can increase risk)
        - HbA1c Level: {hba1c_level} (Ideal range: <5.7%, as higher values can indicate prediabetes or diabetes)
        - Blood Glucose Level: {blood_glucose_level} mg/dL (Ideal range: 70-140 mg/dL fasting, as higher values can indicate diabetes)
        """)
    
    # Make prediction using the model
    input_data = np.array([[gender_dict[gender], age, hypertension_dict[hypertension], heart_disease_dict[heart_disease],
                            smoking_history_dict[smoking_history], bmi, hba1c_level, blood_glucose_level]])
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.success('The model predicts: No diabetes')
        rain( 
            emoji="🎉", 
            font_size=30,  # the size of emoji 
            falling_speed=20,  # speed of raining 
            animation_length="0.5",  # for how much time the animation will happen 
        ) 
    else:
        st.error('The model predicts: Diabetes')
        st.markdown('💔 Please consult with a doctor for advice and guidance.')
        rain( 
            emoji="💔", 
            font_size=30,  # the size of emoji 
            falling_speed=20,  # speed of raining 
            animation_length="0.5",  # for how much time the animation will happen 
        ) 
    # Feature importance graph and summary inside a toggle
with st.expander("View Feature Importance"):
    feature_names = X_train.columns
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(importance_df, x='feature', y='importance', title='Feature Importance')
    fig.update_layout(xaxis_title='Feature', yaxis_title='Importance')
    st.plotly_chart(fig)
    
    # Summary of feature importances
    top_features = importance_df['feature'].iloc[:2].tolist()
    st.markdown(f"The most influential factors in predicting diabetes are **{top_features[0]}** and **{top_features[1]}**.")

    
#Install plotly and streamlit-extras for code to run
