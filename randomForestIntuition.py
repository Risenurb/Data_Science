import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

desired_width = 320
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 20)

data= pd.read_csv(r"C:\Users\risen\Documents\LSBU\DATA_SCIENCE_SELF_LEARNING\MachineLearning with Python\Leacture "
                  r"material\Code_and_Datasets\Part 2 - Regression\Section 9 - Random Forest "
                  r"Regression\Python\Position_Salaries.csv")

x = data.iloc[:,-2].values
y = data.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
rfRegressor = RandomForestRegressor(n_estimators=10,random_state=0)
rfRegressor.fit(x.reshape(len(x),1),y)

print(rfRegressor.predict([[6.5]]))
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color ="red")
plt.plot(x_grid,rfRegressor.predict(x_grid),color ="blue")
plt.title("DT")
plt.xlabel("position")
plt.ylabel("salaries")
plt.legend()
plt.show()




