from flask import Flask, render_template, url_for, request, redirect
import numpy as np
import pickle
import pandas as pd

# Load models
model1 = 'Diabetes_Risk_Model.pkl'
model2 = 'Readmission_Risk_Model.pkl'
with open(model1, 'rb') as f:
    clf1 = pickle.load(f)   
with open(model2, 'rb') as f:
    clf2 = pickle.load(f)   

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')
    
@app.route('/d_predict', methods=['GET', 'POST'])
def d_predict():
    
    if request.method == 'POST':
        # Save our user's info
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        age = int(request.form['age'])

        # Fill blanks with averages
        if glucose == '': glucose = 121
        if bp == '': bp = 69
        if st == '': st = 21
        if insulin == '': insulin = 80
        if bmi == '': bmi = 32
        
        # Run model
        entries = np.array([[preg, glucose, bp, st, insulin, bmi, age]])
        verdict = clf1.predict(entries)
        if verdict[0] == True:
            prediction = 'You share traits similar to those found in people with diabetes'
        else:
            prediction = 'You do not share traits similar to those found in people with diabetes'
        return render_template('d_predict.html', predict = prediction)
    else:
        return render_template('d_predict.html')
    
@app.route('/readmission',  methods=['GET', 'POST'])
def readmission():
    if request.method == 'POST':
        # Save our user's info
        race = request.form['race']
        gender = request.form['gender']
        age = int(request.form['age'])
        diag_1 = request.form['diag_1']
        diag_2 = request.form['diag_2']
        diag_3 = request.form['diag_3']
        number_diagnoses = int(request.form['number_diagnoses'])

        # assign age
        if age < 10:
            age_group = [1,0,0,0,0,0,0,0,0,0]
        elif age < 20:
            age_group = [0,1,0,0,0,0,0,0,0,0]
        elif age < 30:
            age_group = [0,0,1,0,0,0,0,0,0,0]
        elif age < 40:
            age_group = [0,0,0,1,0,0,0,0,0,0]
        elif age < 50:
            age_group = [0,0,0,0,1,0,0,0,0,0]
        elif age < 60:
            age_group = [0,0,0,0,0,1,0,0,0,0]
        elif age < 70:
            age_group = [0,0,0,0,0,0,1,0,0,0]
        elif age < 80:
            age_group = [0,0,0,0,0,0,0,1,0,0]
        elif age < 90:
            age_group = [0,0,0,0,0,0,0,0,1,0]
        else:
            age_group = [0,0,0,0,0,0,0,0,0,1]


        # Encode answers
        # race
        if race == 'Caucasian': ethnicity =       [1,0,0,0,0,0]
        if race == 'AfricanAmerican': ethnicity = [0,1,0,0,0,0]
        if race == 'Asian': ethnicity =           [0,0,0,0,0,1]
        if race == 'Hispanic': ethnicity =        [0,0,0,1,0,0]
        if race == 'Other': ethnicity =           [0,0,0,0,1,0]
        # gender
        if gender == 'male': sex =   [0,1,0]
        if gender == 'female': sex = [1,0,0]
        # diag1
        if diag_1 == 'Diabetes': diagnosis1 =        [1,0,0,0,0,0,0,0,0]
        if diag_1 == 'Neoplasms': diagnosis1 =       [0,1,0,0,0,0,0,0,0]
        if diag_1 == 'Other/None': diagnosis1 =      [0,0,1,0,0,0,0,0,0]
        if diag_1 == 'Circulatory': diagnosis1 =     [0,0,0,1,0,0,0,0,0]
        if diag_1 == 'Respiratory': diagnosis1 =     [0,0,0,0,1,0,0,0,0]
        if diag_1 == 'Injury': diagnosis1 =          [0,0,0,0,0,1,0,0,0]
        if diag_1 == 'Musculoskeletal': diagnosis1 = [0,0,0,0,0,0,1,0,0]
        if diag_1 == 'Digestive': diagnosis1 =       [0,0,0,0,0,0,0,1,0]
        if diag_1 == 'Genitourinary': diagnosis1 =   [0,0,0,0,0,0,0,0,1]
        # diag2
        if diag_2 == 'Diabetes': diagnosis2 =        [0,1,0,0,0,0,0,0,0]
        if diag_2 == 'Neoplasms': diagnosis2 =       [0,0,1,0,0,0,0,0,0]
        if diag_2 == 'Other/None': diagnosis2 =      [1,0,0,0,0,0,0,0,0]
        if diag_2 == 'Circulatory': diagnosis2 =     [0,0,0,1,0,0,0,0,0]
        if diag_2 == 'Respiratory': diagnosis2 =     [0,0,0,0,1,0,0,0,0]
        if diag_2 == 'Injury': diagnosis2 =          [0,0,0,0,0,1,0,0,0]
        if diag_2 == 'Musculoskeletal': diagnosis2 = [0,0,0,0,0,0,1,0,0]
        if diag_2 == 'Digestive': diagnosis2 =       [0,0,0,0,0,0,0,0,1]
        if diag_2 == 'Genitourinary': diagnosis2 =   [0,0,0,0,0,0,0,1,0]
        # diag3
        if diag_3 == 'Diabetes': diagnosis3 =        [0,0,0,1,0,0,0,0,0]
        if diag_3 == 'Neoplasms': diagnosis3 =       [0,1,0,0,0,0,0,0,0]
        if diag_3 == 'Other/None': diagnosis3 =      [1,0,0,0,0,0,0,0,0]
        if diag_3 == 'Circulatory': diagnosis3 =     [0,0,1,0,0,0,0,0,0]
        if diag_3 == 'Respiratory': diagnosis3 =     [0,0,0,0,1,0,0,0,0]
        if diag_3 == 'Injury': diagnosis3 =          [0,0,0,0,0,1,0,0,0]
        if diag_3 == 'Musculoskeletal': diagnosis3 = [0,0,0,0,0,0,0,1,0]
        if diag_3 == 'Digestive': diagnosis3 =       [0,0,0,0,0,0,0,0,1]
        if diag_3 == 'Genitourinary': diagnosis3 =   [0,0,0,0,0,0,1,0,0]

        # append into correct format
        entries2 = []
        entries2.append(ethnicity)
        entries2.append(sex)
        entries2.append(age_group)
        entries2.append(diagnosis1)
        entries2.append(diagnosis2)
        entries2.append(diagnosis3)
        entries2.append(number_diagnoses)
        entries3 = np.array(entries2)
        
        # Run model
        verdict2 = clf2.predict(entries3)
        if verdict2[0] == 1:
            predict = 'Your visit was similar to visits by people who were readmitted within 30 days of their initial hospitalization'
        else:
            predict = 'Your visit was not similar to visits by people who were readmitted within 30 days of their initial hospitalization'
        return render_template('readmission.html', prediction2 = predict)
    else:
        return render_template('readmission.html')

if __name__ == '__main__':
	app.run(debug=True)