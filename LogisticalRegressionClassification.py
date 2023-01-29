import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

desired_width = 320
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 100)

data = pd.read_csv(r"C:\Users\risen\Documents\LSBU\DATA_SCIENCE_SELF_LEARNING\MachineLearning with Python\Leacture "
                   r"material\Code_and_Datasets\Part 3 - Classification\Section 14 - Logistic "
                   r"Regression\Python\Social_Network_Ads.csv")

# print(data)
# selecting indepepndent and dependent variables
x = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

# splitting dataset into training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# features scalling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
# print(x_train)
x_test = sc.fit_transform(x_test)
x_test30 = sc.fit_transform([[30, 87000]])
# print(x_test)

# training logistic regression model

from sklearn.linear_model import LogisticRegression

classifer = LogisticRegression(random_state=0)
classifer.fit(x_train, y_train)
# print(x_test)
print(classifer.predict(x_test30))

###
y_predict = classifer.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_predict.reshape(len(y_predict),1),y_test.reshape(len(y_test),1)),1))

# confusion matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print(confusion_matrix(y_test,y_predict))
print(accuracy_score(y_test,y_predict))

# visualizing training set results


