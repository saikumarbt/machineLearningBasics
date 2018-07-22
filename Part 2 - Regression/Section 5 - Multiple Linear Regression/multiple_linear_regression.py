# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encode the categorical data / variables (State)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

#Create Dummy Encoder for categorized country data transformed into columns
# Here each Dummy Variable column represents one state
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy variable trap to remove the redundant dependency
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting multiple linear regression into the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination.
# Backward Elimination Preparation
# Add a column of 1's to represent b0X where x =1
# Retreive statistical information of Independent variable and remove the independent variable that are not significant for dependent variable (Profit).
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X,  axis=1)
#Start Backward Elimination
# Include all the independent variables first and remove one by one that are not significant.
X_opt = X[:,[0,1,2,3,4,5]] # Orinigal feature of matrix
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Remove independent variable with highest P value above 5%
X_opt = X[:,[0,1,3,4,5]] # Orinigal feature of matrix
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Remove independent variable with highest P value above 5%
X_opt = X[:,[0,3,4,5]] # Orinigal feature of matrix
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Remove independent variable with highest P value above 5%
X_opt = X[:,[0,3,5]] # Orinigal feature of matrix
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Remove independent variable with highest P value above 5%
X_opt = X[:,[0,3]] # Orinigal feature of matrix
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


