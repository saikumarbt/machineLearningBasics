# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 17:39:57 2018
@author: Sai Kumar

"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# Create matrix of features
X=dataset.iloc[:, :-1].values # Matrix of features / independent Variables
Y=dataset.iloc[:,3].values # Extracting Dependent variable vector

# Splitting the dataset into Training Set & Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling (Standardization / Normalization)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Since Y is a classification problem in this case will not be feature scaling.
"""