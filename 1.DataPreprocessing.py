import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
#### 1.Importing dataset
dataset = pd.read_csv(r"C:\Users\risen\Documents\LSBU\DATA_SCIENCE_SELF_LEARNING\MachineLearning with Python\Leacture "
                      r"material\Code_and_Datasets\Part 1 - Data Preprocessing\Section 2 Part 1 - Data "
                      r"Preprocessing\Python\Data.csv")
# print(dataset)
# lets create independent variable feature and dependent variable vector
# take the values of all raws and columns except for the last one as last one acts as dependent variable
X = dataset.iloc[:, :-1].values
print(X)
#### 2. Lets get the values for the last column which is dependent variable
Y = dataset.iloc[:, -1].values
print(Y)

#### 3. Taking care of missing data in the dataset
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
# transform method will do replacement of the missing values
# Second and third columns of X will be replaced by new imputed columns
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

#### 4.Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
print(X)

#### 5. enccoding dependent paprameter y. label encoding to 1 or 0
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y = le.fit_transform(Y)
# print(Y)

#### 6.Splitting the dataset into Training and test
from sklearn.model_selection import train_test_split

X_train, X_test, Y_training, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(len(X_train))
print("Xtrain test ")
print(X_test)
print("Y training")
print(Y_training)
print(Y_test)

#### 7. Feature scalling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)