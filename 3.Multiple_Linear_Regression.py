#### Multiple linear regression
# 5 methods of building models
# -All in - if you know that all your variables contribute to your prediction
# -Backward elimination-
# -Forward selection
# -Bidirectional elimination
# -Score comparison


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

desired_width = 320
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 10)

# Importing file
data = pd.read_csv(r"C:\Users\risen\Documents\LSBU\DATA_SCIENCE_SELF_LEARNING\MachineLearning with Python\Leacture "
                   r"material\Code_and_Datasets\Part 2 - Regression\Section 5 - Multiple Linear "
                   r"Regression\Python\50_Startups.csv")
# print(data.describe())

# Modifying columns names
data.columns = ["R&D_Spend", "Administration", "Marketing_Spend", "State", "Profit"]
# print(data.columns)

# extracting dependent  and independent variables
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# print(x[:, 0:3])
print(x[0])

# Dealing with missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=0, strategy="mean")
x[:, 0:3] = imputer.fit_transform(x[:, 0:3])
# print(x[:, 3])

# encoding categorical values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

column_trans = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
x = column_trans.fit_transform(x)
print(x)

# splitting dataset into training  and test
from sklearn.model_selection import train_test_split

x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(len(x_training))
# Training the Multiple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_training, y_training)

# predicting the test results
y_predict = regressor.predict(x_test)
np.set_printoptions(precision=2)
# concatenating and reshaping the array
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))

# making single prediction
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))
# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)

##Profit=86.6×Dummy State 1−873×Dummy State 2+786×Dummy State 3+0.773×R&D Spend+0.0329×Administration+0.0366×Marketing Spend+42467.53

from sklearn.metrics import r2_score
print(r2_score(y_test, y_predict))