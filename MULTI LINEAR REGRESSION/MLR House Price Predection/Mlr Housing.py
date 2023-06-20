# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\NIT 10AM\Notes\MAR\14th\14th\MLR\House_data.csv")

dataset1 = dataset.drop(['id','date'], axis = 1)
X = dataset1.iloc[:,1:].values
y = dataset1.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
