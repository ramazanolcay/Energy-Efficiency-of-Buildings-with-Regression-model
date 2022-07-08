# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 21:12:31 2022

@author: ramaz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("ENB2012_data.csv")

veriler.columns = ["Relative Compactness","Surface Area","Wall Area",
                "Roof Area", "Overall Height","Orientation","Glazing Area",
                "Glazing Area Distribution", "Heating Load", "Cooling Load"]

nan_check = np.isnan(veriler).any()

x  = veriler.iloc[:,:-2]
X  = x.values
y = veriler.iloc[:,-2:]
Y = y.values


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.33 , random_state =0)


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()

lin_reg1 = LinearRegression()
lin_reg1.fit(x_train,y_train)
y_pred_lin_reg = lin_reg1.predict(x_test)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

r2_score_lr = r2_score(y_pred_lin_reg,y_test)
print("Lineer Reg R2",r2_score_lr)
mse_lr = mean_squared_error(y_test, y_pred_lin_reg)
print("Mean Squared Linear",mse_lr)

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)
y_pred_plr = lin_reg2.predict(poly_reg.fit_transform(x_test))
r2_score_plr = r2_score(y_pred_plr,y_test)
print("Polynomial Reg R2",r2_score_plr)
mse_plr = mean_squared_error(y_test, y_pred_plr)
print("Mean Squared Linear",mse_plr)

#DesicionTree
from sklearn.tree import DecisionTreeRegressor
reg_dt = DecisionTreeRegressor(random_state=0)
reg_dt.fit(x_train,y_train)

y_predict_dt = reg_dt.predict(x_test)
r2_dt = r2_score(y_predict_dt,y_test)
mse_dt = mean_squared_error(y_test, y_predict_dt)
print("DesicionTree R2",r2_dt)
print("Mean Squared Decision Tree",mse_dt)

#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=100,random_state=0)
rf_reg.fit(x_train,y_train)
y_pred_rf = rf_reg.predict(x_test)
r2_score_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print('Random Forest R2',r2_score_rf)
print("Mean Squared Random Forest",mse_rf)

#GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
gpr = GaussianProcessRegressor(random_state=0)
gpr.fit(x_train,y_train)
y_pred_gpr = gpr.predict(x_test)
r2_score_gpr = r2_score(y_test, y_pred_gpr)
mse_gpr = mean_squared_error(y_test, y_pred_gpr)
print('Gaussian Process Regressor R2',r2_score_gpr)
print("Mean Squared Gaussian Process Regressor",mse_gpr)

#PLS Regression
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=8)
pls.fit(x_train,y_train)
y_pred_pls = pls.predict(x_test)
r2_score_pls = r2_score(y_test, y_pred_pls)
mse_pls = mean_squared_error(y_test, y_pred_pls)
print('PLS Regressor R2',r2_score_pls)
print("Mean Squared Error PLS",mse_pls)

#XGboost
import xgboost as xg
xgb_r = xg.XGBRegressor()
xgb_r.fit(x_train, y_train)
pred_xg = xgb_r.predict(x_test)
r2_score_xg = r2_score(y_test, pred_xg)
mse_xg = mean_squared_error(y_test, pred_xg)
print('XGboost Regressor R2',r2_score_xg)
print("Mean Squared Error XGboost",mse_xg)

tips = [
        [mse_lr,r2_score_lr]
        ,[mse_plr,r2_score_plr]
        ,[mse_dt,r2_dt]
        ,[mse_rf,r2_score_rf]
        ,[mse_gpr,r2_score_gpr]
        ,[mse_pls,r2_score_pls]
        ,[mse_xg,r2_score_xg]
        ]
tips = pd.DataFrame(data=tips).transpose()
tips.columns = ["Linear","Polynomial","DecisionTree","RandomForest"
                            ,"Gaussian","PLS","XGboost"]
print(tips)

import seaborn as sbn

sbn.scatterplot(x=tips.iloc[1,:],y = tips.iloc[2,:], data=tips)
