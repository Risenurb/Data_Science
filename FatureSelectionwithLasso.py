import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load the Boston housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit a Lasso Regression model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Get the feature importance scores
importance = np.abs(lasso.coef_)

# Sort the features by importance
sorted_idx = np.argsort(importance)[::-1]

# Print the most important features
for i in sorted_idx:
    print(f"{X.columns[i]}: {importance[i]}")
