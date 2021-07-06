# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:04:03 2020

@author: user
"""

import pandas as pd


#Wczytujemy naszą bazę danych # do wczytywania plików excela:pd.ExcelFile, pd.read_excel
kraje=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab3\\kraje_szacowanie.csv")

#Model1
#Próba ucząca
internet_uzytkownicy_train1=kraje['internet_użytkownicy'][0:87]
internet_uzytkownicy_predict_train1=kraje['internet_użytkownicy_przewidywana1'][0:87]

#MSE i RMSE
from sklearn.metrics import mean_squared_error
MSE_train1=mean_squared_error(internet_uzytkownicy_train1,internet_uzytkownicy_predict_train1)
print('Model 1 próba ucząca')
print('Błąd sredniokwadratowy MSE=', MSE_train1)
import math
RMSE_train1=math.sqrt(MSE_train1)
print('RMSE=',RMSE_train1)

from sklearn.metrics import mean_absolute_error
MAE_train1=mean_absolute_error(internet_uzytkownicy_train1,internet_uzytkownicy_predict_train1)
print('Średni blad bezwzgledny MAE=',MAE_train1)


#Próba testowa
internet_uzytkownicy_test1=kraje['internet_użytkownicy'][87:129]
internet_uzytkownicy_predict_test1=kraje['internet_użytkownicy_przewidywana1'][87:129]

#MSE 
MSE_test1=mean_squared_error(internet_uzytkownicy_test1,internet_uzytkownicy_predict_test1)
print('Model 1 próba testowa')
print('Błąd sredniokwadratowy MSE=', MSE_test1)

RMSE_test1=math.sqrt(MSE_test1)
print('RMSE=',RMSE_test1)

MAE_test1=mean_absolute_error(internet_uzytkownicy_test1,internet_uzytkownicy_predict_test1)
print('Średni blad bezwzgledny MAE=',MAE_test1)


#######################################################################


#Model2
#Próba ucząca
internet_uzytkownicy_train2=kraje['internet_użytkownicy'][0:87]
internet_uzytkownicy_predict_train2=kraje['internet_użytkownicy_przewidywana2'][0:87]

#MSE i RMSE
MSE_train2=mean_squared_error(internet_uzytkownicy_train2,internet_uzytkownicy_predict_train2)
print('Model 2 próba ucząca')
print('Błąd sredniokwadratowy MSE=', MSE_train2)

RMSE_train2=math.sqrt(MSE_train2)
print('RMSE=',RMSE_train2)

MAE_train2=mean_absolute_error(internet_uzytkownicy_train2,internet_uzytkownicy_predict_train2)
print('Średni blad bezwzgledny MAE=',MAE_train2)


#Próba testowa
internet_uzytkownicy_test2=kraje['internet_użytkownicy'][87:129]
internet_uzytkownicy_predict_test2=kraje['internet_użytkownicy_przewidywana2'][87:129]

#MSE i RMSE
print('Model 2 próba testowa')
MSE_test2=mean_squared_error(internet_uzytkownicy_test2,internet_uzytkownicy_predict_test2)
print('Błąd sredniokwadratowy MSE=', MSE_test2)

RMSE_test2=math.sqrt(MSE_test2)
print('RMSE=', RMSE_test2)

MAE_test2=mean_absolute_error(internet_uzytkownicy_test2,internet_uzytkownicy_predict_test2)
print('Blad bezwzgledny MAE=',MAE_test2)
