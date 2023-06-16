#import libraries + packages
import os
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


dashboard_location = os.path.dirname(__file__)

#main app heading - 'Stroke Predictor' 
st.write("""
# Stroke Predictor App
""")


#image displayed on main page 
image = Image.open(os.path.join(dashboard_location,'stroke.jpg'))
st.image(image)


#background information on stroke 
st.write("""
## Background 
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This app is used to predict whether a patient is likely to get stroke based on 11 clinical features for predicting stroke events such as gender, age, various diseases, and smoking status. 
""")


#form header 
st.header('Clinical Determinants of Stroke')

#user input function - collects user input data on stroke features and organizes input data into a df   
with st.form(key='my_form'):
    gender = st.selectbox('Gender',["Male", "Female", "Other"])
    age = st.number_input('Age', 0, 100)
    hypertension = st.selectbox('Hypertension',["Yes", "No"])
    heart_disease = st.selectbox('Heart Disease',["Yes", "No"])
    ever_married = st.selectbox('Ever Married',["Yes", "No"])
    work_type = st.selectbox('Work Type', ["Government", "Never worked", "Private", "Self-employed", "children"])
    Residence_type = st.selectbox('Residence Type', ["Rural", "Urban"])
    avg_glucose_level = st.slider('Average Glucose Level', 80.0, 50.0, 380.0)
    bmi = st.slider('BMI', 24.5 , 15.0, 50.0)
    smoking_status = st.selectbox('Smoking Status', ["smokes", "formerly smoked", "never smoked",  "Unknown"])
    submit = st.form_submit_button(label='Submit')

    if submit:
        data = {'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
        }

        inputs = pd.DataFrame(data, index=[0])

        #displays the feature values from user input
        st.subheader('User Input parameters')
        st.write(inputs)

        #### BUILD ML MODEL ####
        #read in dataset (data contains 5110 observations)
        df = pd.read_csv(os.path.join(dashboard_location,'stroke-dataset.csv'))

        #identify missing data 
        df.isnull().sum()

        #impute missing values (interpolation method - mean/average of column) 
        df = df.dropna()

        #one-hot encoding for categorical features, create a new dummy feature for each unique value in the nominal feature column
        df = pd.get_dummies(df[['gender', 'age', 'hypertension','heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']])

        #split dataset into training and test datasets
        X = df.drop(['stroke'], axis = 1)
        y = df['stroke']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

        #standardize data
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        #train random forest classifier and get accuracy (~94%)
        rf = RandomForestClassifier()
        rf.fit(X_train,y_train)
        y_pred_rf=rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred_rf)

        ########

        st.write('Classifier Accuracy: %.3f' % accuracy) 

        #prep user input
        inputs.loc[inputs['hypertension'] == 'Yes', 'hypertension'] = 1
        inputs.loc[inputs['hypertension'] == 'No', 'hypertension'] = 0
        inputs.loc[inputs['heart_disease'] == 'Yes', 'heart_disease'] = 1
        inputs.loc[inputs['heart_disease'] == 'No', 'heart_disease'] = 0
        inputs.loc[inputs['work_type'] == 'Government', 'work_type'] = "Govt_job"

        #one-hot encoding for categorical features in user input, create a new dummy feature for each unique value in the nominal feature column.
        predict_inputs = pd.get_dummies(inputs[['gender', 'age', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']])
        predict_inputs= predict_inputs.assign(hypertension=[inputs.iloc[0]['hypertension']])
        predict_inputs = predict_inputs.assign(heart_disease=[inputs.iloc[0]['heart_disease']])


        #gender married work type residence type smoking status 
        gender_cols = ['gender_Female', 'gender_Male', 'gender_Other']
        ever_married_cols = ['ever_married_No', 'ever_married_Yes']
        work_type_cols = ['work_type_Govt_job',	'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children']
        Residence_type_cols = ['Residence_type_Rural', 'Residence_type_Urban']
        smoking_status_cols = ['smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']

        cols_lists = [['gender_Female', 'gender_Male', 'gender_Other'],['ever_married_No', 'ever_married_Yes'], ['work_type_Govt_job',	'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children'], ['Residence_type_Rural', 'Residence_type_Urban'],['smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'] ]
        for cols in cols_lists:
            predict_inputs = predict_inputs.reindex(predict_inputs.columns.union(cols, sort=False), axis=1, fill_value=0)

        predict_inputs = predict_inputs[X_train.columns]

        #user input prediction using trained classifier
        prediction = rf.predict(predict_inputs)
        st.write('Stroke prediction: %.3f' % prediction)

        prediction_proba = rf.predict_proba(predict_inputs)

        st.subheader('Prediction Probability')
        st.write(prediction_proba)