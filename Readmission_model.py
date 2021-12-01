import pandas as pd
import numpy as np
import pickle
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

readmission_df = pd.read_csv('Notebooks\diabetic_data.csv')

# Drop: admission_source_id, payer_code, medical_specialty, number_inpatient, number_outpatient, number_emergency, weight, patient_nbr, medications
readmission_cleaned = readmission_df.drop(columns=['admission_source_id', 'payer_code', 'medical_specialty', 'number_inpatient', 'number_outpatient', 'number_emergency', 
                                                  'weight', 'patient_nbr', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 
                                                           'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                                                           'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
                                                           'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed'])


substring = "V"
for i in range (0, len(readmission_cleaned['diag_1'])):
    if substring in readmission_cleaned['diag_1'][i]:
        readmission_cleaned['diag_1'][i]='2'
        
for i in range (0, len(readmission_cleaned['diag_2'])):
    if substring in readmission_cleaned['diag_2'][i]:
        readmission_cleaned['diag_2'][i]='2'
        
for i in range (0, len(readmission_cleaned['diag_3'])):
    if substring in readmission_cleaned['diag_3'][i]:
        readmission_cleaned['diag_3'][i]='2'
        
substring = "?"
for i in range (0, len(readmission_cleaned['diag_1'])):
    if substring in readmission_cleaned['diag_1'][i]:
        readmission_cleaned['diag_1'][i]='2'
        
for i in range (0, len(readmission_cleaned['diag_2'])):
    if substring in readmission_cleaned['diag_2'][i]:
        readmission_cleaned['diag_2'][i]='2'
        
for i in range (0, len(readmission_cleaned['diag_3'])):
    if substring in readmission_cleaned['diag_3'][i]:
        readmission_cleaned['diag_3'][i]='2'

substring = "E"
for i in range (0, len(readmission_cleaned['diag_1'])):
    if substring in readmission_cleaned['diag_1'][i]:
        readmission_cleaned['diag_1'][i]='2'
        
for i in range (0, len(readmission_cleaned['diag_2'])):
    if substring in readmission_cleaned['diag_2'][i]:
        readmission_cleaned['diag_2'][i]='2'
        
for i in range (0, len(readmission_cleaned['diag_3'])):
    if substring in readmission_cleaned['diag_3'][i]:
        readmission_cleaned['diag_3'][i]='2'

for i in range (0, len(readmission_cleaned['diag_1'])):
    readmission_cleaned['diag_1'][i] = float(readmission_cleaned['diag_1'][i])
for i in range (0, len(readmission_cleaned['diag_2'])):
    readmission_cleaned['diag_2'][i] = float(readmission_cleaned['diag_2'][i])
for i in range (0, len(readmission_cleaned['diag_3'])):
    readmission_cleaned['diag_3'][i] = float(readmission_cleaned['diag_3'][i])

bin_names = ['Other','Neoplasms','Diabetes','Neoplasms','Other','Neoplasms','Other','Circulatory','Respiratory','Digestive','Genitourinary'
             ,'Other','Neoplasms','Musculoskeletal','Other','Neoplasms','Circulatory','Respiratory','Digestive','Genitourinary',
             'Neoplasms','Injury']
bins = [0,139,249.99,250.99,279,289,319,389,459,519,579,629,679,709,739,759,784,785,786,787,788,799,1000]
readmission_cleaned['diag_1'] = pd.cut(readmission_cleaned['diag_1'], bins, labels=bin_names, include_lowest=True, ordered = False)
readmission_cleaned['diag_2'] = pd.cut(readmission_cleaned['diag_2'], bins, labels=bin_names, include_lowest=True, ordered = False)
readmission_cleaned['diag_3'] = pd.cut(readmission_cleaned['diag_3'], bins, labels=bin_names, include_lowest=True, ordered = False)

readmission_cleaned['Target'] = readmission_cleaned['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

readmission_cleaned2 = readmission_cleaned.drop(columns=["readmitted", 'encounter_id', 'admission_type_id', 'A1Cresult', 'max_glu_serum', 'discharge_disposition_id'
                                                         ,'num_lab_procedures','time_in_hospital','num_medications','num_procedures'])

ce_OHE = ce.OneHotEncoder(cols=['race','gender', 'age', 'diag_1', 'diag_2', 'diag_3'])

readmission_encoded = ce_OHE.fit_transform(readmission_cleaned2)

y = readmission_encoded.Target.values
X = readmission_encoded.drop(columns="Target").values

X_train, X_test, y_train, y_test = train_test_split(X, y) 

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train_scaled, y_train)

output = open('Readmission_Risk_Model.pkl', 'wb')
pickle.dump(clf, output)