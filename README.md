# Diabetes Prediction App

## Overview
This project is a web-based application designed to predict the likelihood of diabetes using various health parameters. It uses machine learning algorithms to provide a prediction based on user input. The app is built with Streamlit and uses a RandomForestClassifier for prediction.

For a live demonstration, visit the app at [**Diabetes Prediction App**](https://diabetes-prediction-capstone-spillai.streamlit.app/)

For a demo video, check this [**Youtube video**](https://youtu.be/7Mtpg7Ravzc)

Listen to this [**AI Song**](https://youtu.be/of2PEbG8jcc) made for this app here.

## Features
- Interactive sidebar for user input collection.
- Radar chart visualization for comparing health metrics against healthy ranges.
- Feature importance graph to highlight the most significant factors for diabetes prediction.
- Information block with resources and feedback link.
- Customized UI with gradient background and animation effects.
- Downloadable PDF report generation with prediction results.

## How It Works
This Diabetes Prediction App is an interactive tool designed to evaluate the likelihood of diabetes in individuals based on their personal health information. It provides instant predictions, personalized health recommendations, and custom meal planning, along with educational resources for users to engage with. Below is a detailed walkthrough of the app's features and functionalities.

### User Input Collection
- **Interactive Sidebar:** Users are greeted with a sidebar where they can enter personal health metrics such as gender, age, BMI, blood glucose levels, and more.
- **Aesthetically Pleasing Interface:** The sidebar is enhanced with a medical-themed background, providing an engaging experience for users as they input their data.

### Health Risk Assessment
- **Real-time Prediction:** Utilizing a machine learning model, the app calculates the user’s risk of diabetes based on the provided health metrics and displays the outcome immediately.
- **Algorithmic Processing:** The app's backend employs a RandomForestClassifier, a robust algorithm for health risk predictions.

### Personalized Feedback
- **Health Recommendations:** Depending on the prediction results, the app delivers tailored health advice. For example, overweight users receive suggestions for weight management.
- **Meal Plan Suggestions:** The app proposes meal plans that align with the user’s health needs:
  - **For High BMI:** A meal plan rich in high-fiber and low-fat foods is recommended.
  - **For High Blood Glucose:** A meal plan with low-glycemic-index foods is suggested.
  - **For Diabetes Prevention:** A balanced meal plan with nutrient-dense foods is provided to all users.

### Reporting
- **Downloadable PDF Report:** Users can download a personalized report containing their health data, prediction results, and specific recommendations for diet and health. Here is a sample of the [report](https://github.com/sanalpillai/Diabetes-Prediction/blob/main/diabetes_prediction_report.pdf)  generated.
- **Customizable Report Content:** The PDF report includes details like the user's prediction outcome, health recommendations, and meal plans, formatted for clarity and ease of reading.

### Data Visualization
- **Radar Chart Analysis:** The app visualizes the user’s health metrics on a radar chart, comparing them against healthy ranges for an intuitive analysis.
- **Graphical Health Insights:** Visual aids help users quickly understand how their health metrics match up to recommended standards.

### Educational Resources
- **Access to Information:** Links to external resources are provided, guiding users to authoritative websites for further information on diabetes prevention and healthy living.
- **Empowering Users:** The app empowers users with knowledge to take proactive steps toward better health maintenance.

### Technical Insights
- **Feature Importance:** For those interested in the technical aspects, the app provides an insight into which health metrics are most influential in predicting diabetes.
- **Expandable Sections:** Users can expand sections in the app to explore the underlying factors that contribute to the risk assessment.

### Utilization
By providing instant feedback and actionable advice, this Diabetes Prediction App not only informs users about their current health status but also encourages lifestyle adjustments and preventive measures to maintain or improve health outcomes.

### Accessibility and Ease of Use
- **User-Friendly Design:** The app's layout is designed to be navigated effortlessly by users of all ages and backgrounds.
- **Clear Instructions:** Step-by-step instructions ensure users know exactly how to input their data and interpret their results.

### Privacy Consideration
- **Data Privacy:** Users' health data is not stored, ensuring privacy and confidentiality of personal information.

### Interactivity and Engagement
- **Dynamic Feedback:** Celebratory animations are displayed for a 'No diabetes' prediction, while empathetic responses are shown for a diabetes outcome, enhancing user engagement.

### Continuous Learning
- **Model Improvement:** The machine learning model is designed to improve over time, ensuring predictions become more accurate as more data is processed.

### Custom Feature Engineering
- **Tailored Algorithm:** Feature engineering is performed to ensure that the model considers the most relevant factors for diabetes prediction.

The Diabetes Prediction App stands out as a comprehensive tool that blends medical data analysis with user engagement and education. It not only serves as an early warning system for potential health issues but also plays a role in promoting healthier lifestyle choices, all within a privacy-conscious framework. This application is a testament to the power of combining healthcare with modern technology to improve individual well-being.


## Installation
To set up the Diabetes Prediction App locally on your machine, please follow the steps below:

1. Clone the repository:
```bashgit clone https://github.com/yourusername/Diabetes-Prediction-Capstone.git```

2. Change into the project directory:
```bashcd Diabetes-Prediction-Capstone```

3. Install the required dependencies:
```bashpip install -r requirements.txt```

4. Run the app using Streamlit:
```bashstreamlit run diabetes_prediction_app.py```

## Screenshots

*Initial homepage after running the app*
![Initial Homepage](https://github.com/sanalpillai/Diabetes-Prediction/blob/main/Screenshots/Initial%20Homepage.png)

---
*User input via sidebar.*<br>
![Sidebar Input](https://github.com/sanalpillai/Diabetes-Prediction/blob/main/Screenshots/Sidebar.png)

---
*Display of prediction results and profile photo.*
![Prediction Results](https://github.com/sanalpillai/Diabetes-Prediction/blob/main/Screenshots/Personalized%20Health%20Recommendations.png)

---
*Displays meal planning and recipes according to the patient's health*
![Meal Plannig and Recipes](https://github.com/sanalpillai/Diabetes-Prediction/blob/main/Screenshots/Meal%20Planning%20and%20Recipes.png)

---
*Feature importance graph visualized.*
![Feature Importance](https://github.com/sanalpillai/Diabetes-Prediction/blob/main/Screenshots/Feature%20Importance.png)

---
*Button to download the prediction report as a PDF.*
![Download PDF Report](https://github.com/sanalpillai/Diabetes-Prediction/blob/main/Screenshots/Download%20Report.png)

---
*Pdf reported generated after pressing the button*<br>
![PDF Report](https://github.com/sanalpillai/Diabetes-Prediction/blob/main/Screenshots/PDF%20Report.png)

---

## Resources Section
The 'About & Resources' section provides users with additional information and links to resources on diabetes prevention and healthy living. This aids in user education and empowerment.

## Feedback
We value your feedback and contributions to improve the app. Please feel free to report an issue on the [GitHub issues page](https://github.com/sanalpillai/Diabetes-Prediction/issues).

## Acknowledgements
Credit to the contributors, data sources, or third-party resources/tools used in the development of this application.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


