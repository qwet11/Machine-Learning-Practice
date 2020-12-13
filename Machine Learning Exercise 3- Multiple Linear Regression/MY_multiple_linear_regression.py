# -*- coding: utf-8 -*-
"""
Created on Sun Sept 27 2020

Creator: William Moss

Purpose: To practice creating a Multiple Linear Regression Model
"""

#======Preproccessing Data======
import numpy as np
import matplotlib as plt
import pandas as pd

data_set = pd.read_csv("50_Startups.csv")
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder_x = LabelEncoder()
x[:,3] = label_encoder_x.fit_transform(x[:,3])
onehot_encoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough' ) # Leave the rest of the columns untouched   
x = onehot_encoder.fit_transform(x)
x = x[:, 1:] #avoiding the dummy variable trap

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

#======Multiple Regression======
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

#Testing model
print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test))  