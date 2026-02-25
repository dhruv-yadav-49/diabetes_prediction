# 🩺 Diabetes Prediction System

This project is a **Machine Learning–based Diabetes Prediction System** that predicts whether a person is diabetic or not based on medical, lifestyle, and clinical parameters.

The trained model is deployed using a **Flask web application** that allows users to enter health details and get real-time predictions.

---

## 📌 Project Overview

Diabetes is a chronic disease influenced by lifestyle habits, medical history, and biological indicators.  
This project uses supervised machine learning to classify whether a person has diabetes (**YES / NO**) based on multiple health factors.

The final trained model is saved and integrated into a Flask web app for easy usage.

---

## 📂 Project Structure

├── DiabetesPrediction.ipynb # Model training & evaluation

├── rfc-diabetes.pkl # Trained Random Forest model

├── scaler-diabetes.pkl # StandardScaler for feature scaling

├── app.py # Flask application

├── templates/

│ ├── index.html

│ ├── predict.html

│ └── about.html

└── README.md


---

## 🧠 Machine Learning Workflow

1. Data cleaning and preprocessing  
2. Feature selection and engineering  
3. Encoding categorical variables (Smoking Status)  
4. Feature scaling using StandardScaler  
5. Train-test split  
6. Training multiple classification models  
7. Model evaluation using accuracy score  
8. Saving the best model using Pickle  
9. Deploying the model with Flask  

---

## 🤖 Algorithms Used

The following classification algorithms were trained and evaluated in the notebook:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree Classifier  
- Random Forest Classifier  

---

## 🏆 Best Performing Model

After comparing accuracy scores on the test dataset:

✅ **Random Forest Classifier** achieved the **highest accuracy**  
and showed better performance and stability than other models.

👉 **Final model used for prediction:**  
**Random Forest Classifier**

---

## 🛠️ Tools & Technologies Used

### Programming & Libraries
- Python  
- NumPy  
- Pandas  
- Pickle
- Flask  

### Machine Learning
- scikit-learn  

### Web Development
- HTML  
- CSS
---

## 🌐 Web Application Features

- User-friendly form for health data input  
- Real-time diabetes prediction (YES / NO)  
- Uses trained ML model and scaler  
- Flask backend with HTML frontend  

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries
pip install flask numpy pandas scikit-learn

### 2️⃣ Run the Flask Application
python app.py
---

## 📊 Input Features Used for Prediction

- Age
- Gender
- Alcohol consumption per week
- Physical activity minutes per week
- Diet score
- Sleep hours per day
- Screen time hours per day
- Family history of diabetes
- Hypertension history
- Cardiovascular history
- BMI
- Waist-to-hip ratio
- Systolic blood pressure
- Diastolic blood pressure
- Heart rate
- Total cholesterol
- HDL cholesterol
- LDL cholesterol
- Triglycerides
- Fasting glucose
- Postprandial glucose
- Insulin level
- HbA1c
- Smoking status
---

## 🚀 Future Improvements

- Add model explainability (SHAP / feature importance)
- Improve UI using Bootstrap
- Add cloud deployment (AWS / Render)
- Store prediction history using a database
