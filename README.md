# Diabetes Prediction App

## Overview
This project is a web-based application designed to predict the likelihood of diabetes using various health parameters. It uses machine learning algorithms to provide a prediction based on user input. The app is built with Streamlit and uses a RandomForestClassifier for prediction.

## Features
- Interactive sidebar for user input collection.
- Radar chart visualization for comparing health metrics against healthy ranges.
- Feature importance graph to highlight the most significant factors for diabetes prediction.
- Information block with resources and feedback link.
- Customized UI with gradient background and animation effects.

## How It Works
Users enter their health details, including gender, age, hypertension status, heart disease status, smoking history, BMI, HbA1c level, and blood glucose level. Upon submission, the application processes the inputs through a pre-trained machine learning model to predict whether the user is likely to have diabetes. Results are accompanied by animations to enhance the user experience.

## Installation
Instructions on setting up the project locally. Include steps for cloning the repository, installing dependencies, and running the Streamlit app.

git clone https://github.com/yourusername/Diabetes-Prediction-Capstone.git
cd Diabetes-Prediction-Capstone
pip install -r requirements.txt
streamlit run diabetes_prediction_app.py

vbnet
Copy code

## Usage
Provide a step-by-step guide on how to use the app, possibly with screenshots.

![Sidebar Input](path-to-screenshot.png)
*User input via sidebar.*

![Prediction Results](path-to-screenshot.png)
*Display of prediction results and profile photo.*

![Feature Importance](path-to-screenshot.png)
*Feature importance graph visualized.*

## Resources Section
Discuss the 'About & Resources' section, which provides users with additional information and links to resources on diabetes prevention and healthy living. Mention how it aids in user education.

![About & Resources](path-to-screenshot.png)
*About & Resources block.*

## Feedback
Encourage users to give feedback and contribute to the project by providing a contact email or a link to the issue tracker.

## Acknowledgements
Credit any contributors, data sources, or third-party resources/tools used.

## License
State the license under which the project is released.

Remember to replace `path-to-screenshot.png` with the actual file paths of your screens

App Link: https://diabetes-prediction-spillai.streamlit.app/
