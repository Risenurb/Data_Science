####Simple linear regression
# 1. Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Setting up max number of rows and columns to be displayed
desired_width = 320
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 20)
# 3. importing the dataset
data = pd.read_csv(r"C:\Users\risen\Documents\LSBU\DATA_SCIENCE_SELF_LEARNING\MachineLearning with Python\Leacture "
                   r"material\Code_and_Datasets\Part 2 - Regression\Section 4 - Simple Linear "
                   r"Regression\Python\Salary_Data.csv")
# print(data.info())
# print(data.describe())

# 4. creating arrays of independent and dependent variables
x = data.iloc[:, :-1].values
print(f" The value of independent variables are: {x}")
y = data.iloc[:, -1].values
print(y.shape)
# 5. Splitting dataset into training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(len(x_train))
print(len(x_test))

# 6. training simple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
# 7. predicting results
y_pred = regressor.predict(x_test)
print(y_pred)
print(regressor.predict(x_train).shape)
# 8. visualizing the training dataset
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="green")
plt.xlabel("Year of experience , (Training dataset)")
plt.ylabel("Salaries")
plt.title("Salary vs Experience")
plt.legend()
plt.show()

# 9. visualizing test dataset
plt.scatter(x_test, y_test, color="red")
plt.plot(x_test, regressor.predict(x_test), color="green")
plt.xlabel("Year of experience , (Test data et)")
plt.ylabel("Salaries")
plt.title("Salary vs Experience")
plt.legend()
plt.show()
# predicitng salaries from 11 to 20 years of experience
for i in range(11, 20):
    print(regressor.predict([[i]]))
# printing out intercept and coefficient
print(regressor.intercept_)
print(regressor.coef_)
# So linear regression equation will be
# Salary=9345.94×YearsExperience+26816.19




