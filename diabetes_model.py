import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler



diabetes_df = pd.read_csv('Notebooks\diabetes.csv')

nonzero_df = diabetes_df.loc[list((diabetes_df.Glucose > 0) & 
                                  (diabetes_df.BloodPressure > 0) & 
                                  (diabetes_df.BMI > 0)), :,]

X = nonzero_df.drop(['DiabetesPedigreeFunction','Outcome'], axis=1)
y = nonzero_df['Outcome'] != 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=800, max_depth =90, min_samples_split =2, min_samples_leaf =4, max_features = 'sqrt').fit(X_train_scaled, y_train)

output = open('Diabetes_Risk_Model.pkl', 'wb')
pickle.dump(clf, output)