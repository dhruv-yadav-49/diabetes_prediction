from flask import Flask,render_template,request,redirect,session
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)

with open("rfc-diabetes.pkl","rb") as f:
    model=pickle.load(f)

with open("scaler-diabetes.pkl","rb") as f:
    ss=pickle.load(f)


def predict_diabetes(age=27,gender=1,alcohol_consumption_per_week=0,physical_activity_minutes_per_week=105,diet_score=6.20,sleep_hours_per_day=5.60,
                     screen_time_hours_per_day=7.50,family_history_diabetes=1,hypertension_history=0,cardiovascular_history=0,bmi=25.40,waist_to_hip_ratio=0.84,
                     systolic_bp=90,diastolic_bp=76,heart_rate=77,cholesterol_total=200,hdl_cholesterol=73,ldl_cholesterol=96,triglycerides=44,glucose_fasting=132,
                     glucose_postprandial=163,insulin_level=2.15,hba1c=7.18,smoking_status='Former'):
    
    # 1-23 Original numeric features
    temp_array = [
        age, gender, alcohol_consumption_per_week, physical_activity_minutes_per_week,
        diet_score, sleep_hours_per_day, screen_time_hours_per_day, family_history_diabetes,
        hypertension_history, cardiovascular_history, bmi, waist_to_hip_ratio,
        systolic_bp, diastolic_bp, heart_rate, cholesterol_total, hdl_cholesterol,
        ldl_cholesterol, triglycerides, glucose_fasting, glucose_postprandial,
        insulin_level, hba1c
    ]

    # 24-26 Smoking status dummies (Current, Former, Never)
    if smoking_status == 'Current':
        temp_array += [1, 0, 0]
    elif smoking_status == 'Former':
        temp_array += [0, 1, 0]
    else: # Never
        temp_array += [0, 0, 1]
    
    temp_array = np.array([temp_array])
    print("Input array shape:", temp_array.shape)
    
    feature_names = ['Age', 'gender', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
                     'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'family_history_diabetes', 
                     'hypertension_history', 'cardiovascular_history', 'bmi', 'waist_to_hip_ratio', 
                     'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 
                     'ldl_cholesterol', 'triglycerides', 'glucose_fasting', 'glucose_postprandial', 
                     'insulin_level', 'hba1c', 'smoking_status_Current', 'smoking_status_Former', 'smoking_status_Never']
    
    temp_df = pd.DataFrame(temp_array, columns=feature_names)
    temp_array_scaled = ss.transform(temp_df)
    pred = model.predict(temp_array_scaled)
    print("Prediction:", pred)
    if pred[0]==1:
        result='YES'
    else:
        result='NO'

    return result

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/predict",methods=['POST','GET'])

def predict():
    if request.method=='POST':
        age=int(request.form.get('age'))
        gender=float(request.form.get('gender'))
        alcohol_consumption_per_week=int(request.form.get('alcohol_consumption_per_week'))
        physical_activity_minutes_per_week=int(request.form.get('physical_activity_minutes_per_week'))
        diet_score=float(request.form.get('diet_score'))
        sleep_hours_per_day=float(request.form.get('sleep_hours_per_day'))
        screen_time_hours_per_day=float(request.form.get('screen_time_hours_per_day'))
        family_history_diabetes=int(request.form.get('family_history_diabetes'))
        hypertension_history=int(request.form.get('hypertension_history'))
        cardiovascular_history=int(request.form.get('cardiovascular_history'))
        bmi=float(request.form.get('bmi'))
        waist_to_hip_ratio=float(request.form.get('waist_to_hip_ratio'))
        systolic_bp=int(request.form.get('systolic_bp'))
        diastolic_bp=int(request.form.get('diastolic_bp'))
        heart_rate=int(request.form.get('heart_rate'))
        cholesterol_total=int(request.form.get('cholesterol_total'))
        hdl_cholesterol=int(request.form.get('hdl_cholesterol'))
        ldl_cholesterol=int(request.form.get('ldl_cholesterol'))
        triglycerides=int(request.form.get('triglycerides'))
        glucose_fasting=int(request.form.get('glucose_fasting'))
        glucose_postprandial=int(request.form.get('glucose_postprandial'))
        insulin_level=float(request.form.get('insulin_level'))
        hba1c=float(request.form.get('hba1c'))
        smoking_status=request.form.get('smoking_status')

        diabetes=predict_diabetes(age=age,gender=gender,alcohol_consumption_per_week=alcohol_consumption_per_week,physical_activity_minutes_per_week=physical_activity_minutes_per_week,diet_score=diet_score,sleep_hours_per_day=sleep_hours_per_day,
                     screen_time_hours_per_day=screen_time_hours_per_day,family_history_diabetes=family_history_diabetes,hypertension_history=hypertension_history,cardiovascular_history=cardiovascular_history,bmi=bmi,waist_to_hip_ratio=waist_to_hip_ratio,
                     systolic_bp=systolic_bp,diastolic_bp=diastolic_bp,heart_rate=heart_rate,cholesterol_total=cholesterol_total,hdl_cholesterol=hdl_cholesterol,ldl_cholesterol=ldl_cholesterol,triglycerides=triglycerides,glucose_fasting=glucose_fasting,
                     glucose_postprandial=glucose_postprandial,insulin_level=insulin_level,hba1c=hba1c,smoking_status=smoking_status)
        print(diabetes)
        return render_template('predict.html',prediction=diabetes)
    
    return render_template('predict.html')


if __name__=='__main__':
    app.run(debug=True,port=8000,host='0.0.0.0')