import streamlit as st
import numpy as np
import pandas as pd
import fpdf
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from streamlit_extras.stoggle import stoggle
from streamlit_extras.let_it_rain import rain 
from fpdf import FPDF



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
df = pd.read_csv("https://raw.githubusercontent.com/sanalpillai/Diabetes-Prediction/main/Dataset/diabetes_prediction_dataset.csv")
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

class PDF(FPDF):
    def __init__(self):  # Add this __init__ method if not present
        super().__init__()
        self.unifontsubset = True  # Ensure this attribute is set
        
    def header(self):
        # Ensure the right position at the start of each page
        self.set_y(10)
        # Logo
        self.image('https://raw.githubusercontent.com/sanalpillai/Diabetes-Prediction/main/Assets/medical-report-logo.png', 10, 8, 33)
        # Move to the right
        self.cell(80)
        # Set the position for the title
        self.set_x(10)
        self.set_font('Arial', 'B', 18)
        # Title
        self.cell(0, 10, 'Diabetes Prediction Report', 0, 1, 'C')
        # Ensure a sufficient gap after the header
        self.ln(10)

    def chapter_title(self, num, label):
        # Arial 12
        self.set_font('Arial', '', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, 'Chapter %d : %s' % (num, label), 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_body(self, body):
        # Read text file
        # Times 12
        self.set_font('Times', '', 12)
        # Output justified text
        self.multi_cell(0, 10, body)
        # Line break
        self.ln()
        # Mention in italics
        self.set_font('', 'I')
        self.cell(0, 10, '(end of excerpt)')

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')
    
    def check_page_break(self, threshold):
        if self.get_y() + threshold > self.page_break_trigger:
            self.add_page()
    
def create_pdf_report(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level, prediction, health_recommendations, meal_plans):
    pdf = PDF()
    pdf.add_page()
    # Before adding new content, check the y position
    y_before = pdf.get_y()
    if y_before < 40:  # If less than your desired top margin, set the position lower
        pdf.set_y(40)
    

    # Print the chapters
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0,7, 'Patient Information', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    
    patient_info = f"""
    Gender: {gender}
    Age: {age}
    Hypertension: {hypertension}
    Heart Disease: {heart_disease}
    Smoking History: {smoking_history}
    BMI: {bmi}
    HbA1c Level: {hba1c_level}
    Blood Glucose Level: {blood_glucose_level}
    """
    pdf.multi_cell(0, 7, patient_info)
    
    # Line break with a full-width line
    pdf.ln(3)  # Move below the content
    pdf.set_draw_color(0, 0, 0)  # Set the line color to black
    pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())  # Draw the line 
    pdf.ln(3)  # Space after line

    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Prediction Outcome', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    outcome = 'No diabetes' if prediction[0] == 0 else 'Diabetes'
    pdf.cell(0, 7, f"Prediction: {outcome}", 0, 1, 'C')

    # Line break with a full-width line
    pdf.ln(5)  # Move below the content
    pdf.set_draw_color(0, 0, 0)  # Set the line color to black
    pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())  # Draw the line 
    pdf.ln(5)  # Space after line

    # New section for Health Recommendations
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Health Recommendations', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    for recommendation in health_recommendations:  # Use the passed parameter
        pdf.multi_cell(0, 7, recommendation)
        pdf.ln(2)  # Add a small space between recommendations for readability

    # Include meal plans in the PDF
    if meal_plans:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 7, 'Meal Planning and Recipes', 0, 1, 'C')
        pdf.ln(10)  # Add a space after the section title
        pdf.set_font('Arial', '', 12)
        for plan in meal_plans:
            # Check for page break
            pdf.check_page_break(10)  # Adjust the value according to your needs
            # Bold recipe title
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 7, plan['title'], 0, 1, 'L')
            pdf.ln(2)

            # Regular text for description
            pdf.set_font('Arial', '', 12)
            pdf.multi_cell(0, 7, plan['description'])
            pdf.ln(2)

            # List each recipe
            for recipe in plan['recipes']:
                pdf.set_font('Arial', 'B', 12)  # Make recipe names bold
                pdf.cell(0, 7, f"- {recipe}", 0, 1)
            # Line break with a full-width line
            pdf.ln(2)  # Move below the content
            pdf.set_draw_color(0, 0, 0)  # Set the line color to black
            pdf.line(5, pdf.get_y(), pdf.w - 5, pdf.get_y())  # Draw the line 

    return pdf.output(dest='S').encode('latin1')


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
# Main layout with three columns: main content, spacer, and right-hand information block
main_col, spacer, info_col = st.columns([3, 1, 1])

with main_col:
    # Your main app content here
    st.write("# Diabetes Prediction App")
    st.markdown("""
    This app predicts the likelihood of diabetes based on health parameters.
    Enter the required information in the sidebar and press 'Submit' to see the prediction.
    """)
    # Assume the prediction display and other main content goes here

with info_col:
    # Use the st.markdown to apply the custom styles to this container
    st.markdown(
        """
        <div style="background-color: #800080; color: white; padding: 20px; border-radius: 10px; 
                    width: 250px; height: auto; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <h2 style="margin-top: 0;">About & Resources</h2>
            <ul style="padding-left: 20px;"> <!-- Adjusted padding for bullets -->
                <li style="margin-bottom: 10px;"><a href="https://www.cdc.gov/diabetes/prevent-type-2/index.html" target="_blank" style="color: #FFD700; text-decoration: underline;">Diabetes Prevention</a></li> <!-- Added text-decoration -->
                <li><a href="https://www.who.int/news-room/fact-sheets/detail/healthy-diet" target="_blank" style="color: #FFD700; text-decoration: underline;">Healthy Living</a></li> <!-- Added text-decoration -->
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )



# Define the path to your images
image_paths = {
    'Male': 'https://raw.githubusercontent.com/sanalpillai/Diabetes-Prediction/main/Assets/Male.png',
    'Female': 'https://raw.githubusercontent.com/sanalpillai/Diabetes-Prediction/main/Assets/Female.png',
    'Other': 'https://raw.githubusercontent.com/sanalpillai/Diabetes-Prediction/main/Assets/Other.png'
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
    st.write("## Patient Report")
    # Display patient results and profile photo side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Display the patient's profile photo
        st.image(image_paths[gender], width=300)
    
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
    # Function to generate personalized health recommendations
    def generate_health_recommendations(prediction, bmi, hba1c_level, blood_glucose_level):
        recommendations = []

        # General advice for all users
        recommendations.append("Regular physical activity can help prevent diabetes and manage blood sugar levels.")
        recommendations.append("A balanced diet rich in fruits, vegetables, and whole grains is beneficial for overall health.")

        # Custom recommendations based on prediction and health metrics
        if prediction[0] == 1:  # If the model predicts diabetes
            recommendations.append("Consider consulting with a healthcare professional for further assessment and guidance.")
            if bmi >= 25:
                recommendations.append("Aiming for a healthy weight can reduce your risk of diabetes. Consider speaking with a dietitian for personalized advice.")
            if hba1c_level >= 5.7:
                recommendations.append("An HbA1c level of 5.7% or higher indicates prediabetes. Discuss with your doctor about monitoring your blood sugar levels more closely.")
            if blood_glucose_level >= 140:
                recommendations.append("High fasting blood glucose levels can be a sign of diabetes. It's important to consult with your doctor for a comprehensive evaluation.")
        else:
            recommendations.append("Maintaining a healthy lifestyle can help keep your risk of diabetes low. Regular check-ups with your doctor are also recommended.")

        return recommendations

    def generate_meal_plans(bmi, blood_glucose_level):
        meal_plans = []

        # For users needing to manage weight
        if bmi >= 25:
            meal_plans.append({
                'title': 'Balanced Weight Management Meal Plan',
                'description': 'A carefully curated meal plan focusing on high-fiber, low-fat foods to support weight loss while ensuring you stay full and satisfied. Key ingredients include lean proteins, whole grains, and plenty of fruits and vegetables to provide balanced nutrition.',
                'recipes': [
                    'Breakfast: Avocado and egg toast on whole-grain bread',
                    'Lunch: Quinoa salad with chickpeas, cucumber, tomatoes, and feta cheese',
                    'Dinner: Grilled salmon with a side of roasted Brussels sprouts and sweet potato',
                    'Snack: Greek yogurt with a handful of walnuts and honey'
                ]
            })

        # For users with higher blood glucose levels
        if blood_glucose_level >= 140:
            meal_plans.append({
                'title': 'Low-Glycemic Index Meal Plan',
                'description': 'Designed to minimize spikes in blood sugar levels, this meal plan includes foods that have a low glycemic index. Expect meals rich in proteins and healthy fats, along with complex carbohydrates that are digested slowly, helping to maintain steady blood sugar levels throughout the day.',
                'recipes': [
                    'Breakfast: Steel-cut oatmeal topped with cinnamon and fresh berries',
                    'Lunch: Lentil and vegetable stew with a side of mixed greens salad',
                    'Dinner: Chicken stir-fry with broccoli, bell peppers, and cashews served over brown rice',
                    'Snack: Sliced apples with almond butter'
                ]
            })

        # General healthy eating plan for diabetes prevention
        meal_plans.append({
            'title': 'Diabetes Prevention Meal Plan',
            'description': 'This general health-focused meal plan is designed for overall well-being and diabetes prevention. It includes a variety of nutrient-dense, antioxidant-rich foods to support healthy blood sugar levels and reduce inflammation.',
            'recipes': [
                'Breakfast: Greek yogurt with mixed berries and a sprinkle of chia seeds',
                'Lunch: Turkey and avocado wrap with whole-grain tortilla and mixed vegetable sticks',
                'Dinner: Baked cod with a lemon-herb crust, served with quinoa and steamed green beans',
                'Snack: A handful of mixed nuts and a piece of fresh fruit'
            ]
        })

        return meal_plans

# Ensure the generate_meal_plans function is correctly called within the main logic of your application


    # Assuming 'prediction', 'bmi', 'hba1c_level', and 'blood_glucose_level' are defined from your model and user inputs
    # Call the function to get personalized recommendations
    recommendations = generate_health_recommendations(prediction, bmi, hba1c_level, blood_glucose_level)

    # Display the recommendations
    st.write("## Personalized Health Recommendations")
    for recommendation in recommendations:
        st.write("- ", recommendation)

    # Assuming 'bmi' and 'blood_glucose_level' are defined
    meal_plans = generate_meal_plans(bmi, blood_glucose_level)

    if meal_plans:
        st.write("## Meal Planning and Recipes")
        for plan in meal_plans:
                st.write(f"### {plan['title']}")
                st.write(plan['description'])
                for recipe in plan['recipes']:
                    st.write(f"- {recipe}")

        
    # Create the PDF
    pdf_content = create_pdf_report(
        gender=gender,
        age=age,
        hypertension=hypertension,
        heart_disease=heart_disease,
        smoking_history=smoking_history,
        bmi=bmi,
        hba1c_level=hba1c_level,
        blood_glucose_level=blood_glucose_level,
        prediction=prediction,
        health_recommendations=recommendations,
        meal_plans=meal_plans
    )

    # Convert to a bytes object
    pdf_bytes = bytes(pdf_content)

    # Use Streamlit's download button to offer the PDF to the user
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="diabetes_prediction_report.pdf",
        mime="application/pdf",
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

    
#Install plotly, fpdf and streamlit-extras for code to run
