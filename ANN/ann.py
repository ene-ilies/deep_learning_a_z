#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 20:42:08 2018

@author: bogdan-ilies
"""

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])

labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Homework
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $60000
# Number of Products: 2
# Does this customer have a credit card ? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $50000
new_rec = [[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 5000]]
new_rec = sc.transform(new_rec)

new_pred = classifier.predict(new_rec)
new_pred = (new_pred > 0.5)

# cross val score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
   classifier = Sequential()
   classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
   classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
   classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
   classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1)
mean = accuracies.mean()
std = accuracies.std()

#binary search cross validation
def build_classifier_params(optimizer='adam', init='uniform'):
   classifier = Sequential()
   classifier.add(Dense(output_dim=6, init=init, activation='relu', input_dim=11))
   classifier.add(Dense(output_dim=6, init=init, activation='relu'))
   classifier.add(Dense(output_dim=1, init=init, activation='sigmoid'))
   classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
   return classifier

from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn=build_classifier_params)

grid_params = {'batch_size': [30, 40], 'nb_epoch': [100, 300], 'optimizer': ['adam', 'rmsprop']}
grid = GridSearchCV(estimator=classifier, param_grid=grid_params, scoring='accuracy', cv=10, n_jobs=1)
grid = grid.fit(X_train, y_train)

