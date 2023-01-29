import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

desired_width = 320
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 20)
dataset = pd.read_csv(r"C:\Users\risen\Documents\LSBU\DATA_SCIENCE_SELF_LEARNING\MachineLearning with Python\Leacture "
                      r"material\Code_and_Datasets\Part 2 - Regression\Section 7 - Support Vector Regression ("
                      r"SVR)\Python\Position_Salaries.csv")

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# turning y into 2D array
y = y.reshape(len(y), 1)

# Feature scalling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# training SVR model
from sklearn.svm import SVR

regressor = SVR(kernel="rbf")
regressor.fit(x, y)
# predicting new result
print((regressor.predict(x).reshape(len(y), 1)).shape)
sc_y.inverse_transform(regressor.predict(x).reshape(len(y), 1))
#print(sc_y.inverse_transform([regressor.predict(sc_x.transform([[6.5]]))]))

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y))
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(len(y), 1)), color="blue")
plt.title(" Support Vector Regression")
plt.xlabel("Position")
plt.ylabel("Salaries")
plt.show()
