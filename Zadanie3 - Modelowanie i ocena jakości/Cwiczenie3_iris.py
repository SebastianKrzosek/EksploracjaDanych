# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:04:03 2020

@author: user
"""

import pandas as pd


#Wczytujemy naszą bazę danych # do wczytywania plików excela:pd.ExcelFile, pd.read_excel
iris=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab3\\iris_klasyfikacja.csv")

#Próba ucząca
klasa_train=iris['klasa'][0:103]
klasa_predict_train=iris['Predicted'][0:103]

#Tworzymy macierz pomyłek dla próby uczącej
from sklearn.metrics import confusion_matrix
macierz_pomyłek1 = confusion_matrix(klasa_train,klasa_predict_train)

#Próba testowa
klasa_test=iris['klasa'][103:150]
klasa_predict_test=iris['Predicted'][103:150]

#Tworzymy macierz pomyłek dla próby testowej
macierz_pomyłek2 = confusion_matrix(klasa_test,klasa_predict_test)





