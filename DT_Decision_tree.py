# Regression trees

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

desired_width = 320
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 20)

data = pd.read_csv(r"C:\Users\risen\Documents\LSBU\DATA_SCIENCE_SELF_LEARNING\MachineLearning with Python\Leacture "
                   r"material\Code_and_Datasets\Part 2 - Regression\Section 8 - Decision Tree "
                   r"Regression\Python\Position_Salaries.csv")

x = data.iloc[:, -2].values
y = data.iloc[:, -1].values
#print(y.reshape(10,1))

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=0)
# training the DT model
dt = dt.fit(x.reshape(10, 1), y)
# using the trained model object we predict the salary of the employee having levels 6.5 and 6.7
print(dt.predict([[6.5],[6.7]]))

# visualizing DT results

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color ="red")
plt.plot(x_grid,dt.predict(x_grid),color ="blue")
plt.title("DT")
plt.xlabel("position")
plt.ylabel("salaries")
plt.legend()
plt.show()