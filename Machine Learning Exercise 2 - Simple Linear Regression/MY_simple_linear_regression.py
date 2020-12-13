"""
Created on Wed Sept 9 2020

Creator: William Moss

Purpose: To practice creating a Simple Linear Regression Model
"""

#======Preproccessing Data======

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv("Salary_Data.csv")
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)



#======Linear Regression======
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting the Simple Linear Regression model to our (train) data
regressor.fit(x_train, y_train)

#Prediction of Test and Training set result
y_pred  = regressor.predict(x_test)
x_pred = regressor.predict(x_train)

#======Visualize the Training Set results======
plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, x_pred, color="red")
plt.title("Salary vs Experience (Training Dataset)")
plt.xlabel("Years in Experience")
plt.ylabel("Salary (in Rupees)")

plt.show()

#======Visualize the Testing Set results======
plt.scatter(x_test, y_test, color="blue")
plt.plot(x_train, x_pred, color="red")
plt.title("Salary vs Experience (Testing Dataset)")
plt.xlabel("Years in Experience")
plt.ylabel("Salary (in Rupees)")