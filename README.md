## Calories Burned Prediction ML App

A machine learning-powered web application that predicts the number of calories burned during a workout based on physiological and workout-related inputs like gender, age, height, weight, duration, heart rate, and body temperature.

Built using:
-  Random Forest Regressor (Scikit-learn)
-  Streamlit (for frontend UI)
-  Google Colab (for model training)
-  GitHub (for version control)

## About

Our project aims to help users track their fitness by estimating how many calories they burn during exercise. By inputting simple body and workout metrics, users can get quick and fairly accurate calorie estimates using a trained ML model.

We used **Random Forest Regression**, a supervised machine learning technique that captures both linear and nonlinear relationships between variables. This model is more robust than linear regression for complex datasets like fitness data, where multiple factors interact.

This application was designed to:
- Encourage user engagement in personal fitness tracking
- Showcase end-to-end ML deployment using Streamlit
- Demonstrate model training, evaluation, and integration into a real-world tool


### Dataset
- Source: Kaggle
- Features used: `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`
- Derived Features: `Height_m`, `BMI`
- Target: `Calories`

### Steps Followed

1. **Data Collection**  
   - Used a public dataset from Kaggle.
2. **Data Preprocessing**
   - Encoded `Gender` (Male = 1, Female = 0)
   - Created `Height_m` and `BMI` features
   - Scaled features using `StandardScaler`
3. **Exploratory Data Analysis (EDA)**
   - Used Seaborn/Matplotlib for correlations and distributions
4. **Model Training**
   - Used `RandomForestRegressor` for flexibility and accuracy
5. **Model Evaluation**
   - Evaluated using:
     - Mean Absolute Error (MAE)
     - R² Score
6. **Deployment**
   - Built a responsive UI with Streamlit
   - Model and scaler saved using `joblib` and integrated into app

