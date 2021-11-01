import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump, load
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

dataset = pd.read_csv('pose2.csv')

X = dataset.iloc[:, 0:80].values
y = dataset.iloc[:, 80].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestClassifier(random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print(accuracy_score(y_test, y_pred))

dump(regressor, 'model.joblib')