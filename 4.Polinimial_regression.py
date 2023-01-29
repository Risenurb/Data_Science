import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
desired_width = 320
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 20)

# 1.importing the dataset
data = pd.read_csv(r"C:\Users\risen\Documents\LSBU\DATA_SCIENCE_SELF_LEARNING\MachineLearning with Python\Leacture "
                   r"material\Code_and_Datasets\Part 2 - Regression\Section 6 - Polynomial "
                   r"Regression\Python\Position_Salaries.csv")

# print(data.head())
# print(data.info())
# print(data.describe())
# 2. Extracting dependent and independent variables
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

print(x[:, 0])
# print([type(y), y.reshape(len(y), 1)])
# print(y)

# 2. As there are no missing values in this dataset we would directly go to the
# next step which is encoding categorical values into unit vectors

# Building linear regression first

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)

# buinding polynomial linear regression model
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

# visualizing linear and polynimial regressions

plt.scatter(x, y, color="red")
plt.plot(x, lin_reg.predict(x), color="blue")
plt.plot(x,lin_reg2.predict(x_poly), color="green")
plt.xlabel("Level")
plt.ylabel("Salaries")
plt.title("Truth or bluff(Linear Regression Model)")
plt.show()
plt.legend()
x_poly6 = poly_reg.fit_transform([[6.5]])
print(lin_reg2.predict(x_poly6))

from sklearn.metrics import r2_score
print(r2_score(y_test, y_predict))