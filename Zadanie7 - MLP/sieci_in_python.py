# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:06:11 2020

@author: user
"""

import pandas as pd

#Wczytujemy naszą bazę danych
churn=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab7\\churn_pl_dane.csv")



#Predyktory jako X
X_names_predyktory=["Czaswspolpracy","Liczbawiadomosci","Dzienminuty","Dzienrozmowy","Wieczorminuty","Wieczorrozmowy","Nocminuty","Nocrozmowy","Miedzynarodoweminuty","Miedzynarodowerozmowy","LiczbarozmowzBOK","Planmiedzy01"]
X = churn[X_names_predyktory]

#Zmienna celu jako y
y=churn['Rezygnacja']


#Dzielimy na zbiór uczący (70%) i zbiór testowy (30%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)


######################################
#Noc minuty z próby testowej przed normalizacja, przyda się do problemu szacowania
y_test2=X_test['Nocminuty']
######################################


#stardaryzacja min - max
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train_norm=scaler.transform(X_train)

X_test_norm=scaler.transform(X_test)


#Budowa sieci 
from sklearn.neural_network import MLPClassifier
siec_neur = MLPClassifier(hidden_layer_sizes=(8,), activation='tanh',solver='lbfgs', alpha=0.0001, max_iter = 1000, random_state=123)
#Uczenie sieci
siec_train=siec_neur.fit(X_train_norm,y_train)




#Przewidywane wartości i przewidywane prawdopodobieństwo
y_predicted=siec_neur.predict(X_test_norm)  
probabilities=siec_neur.predict_proba(X_test_norm)

#Wyznaczmy macierz pomyłek i pozostałe miary jakości
from sklearn.metrics import confusion_matrix
macierz_pomyłek=confusion_matrix(y_test,y_predicted)
tn, fp, fn, tp = confusion_matrix(y_test,y_predicted).ravel()
print('Czułość: ', round(tp/(fn+tp),3))
print('Specyficzność: ', round(tn/(tn+fp),3))
print('Wskaźnik fałszywie negatywnych: ', round(fn/(fn+tp),3))
print('Wskaźnik fałszywie pozytywnych: ', round(fp/(tn+fp),3))
print('Precyzja: ', round(tp/(fp+tp),3))
print('Trafność: ', round((tp+tn)/(tp+fp+tn+fn),3))


#Krzywa ROC     
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
#Funkcja `roc_curve` oblicza wartości potrzebne do narysowania wykresu (ale nie rysuje wykresu). 
#Funkcja ta działa tylko w przypadku klasyfikacji binarnej. Jako parametry podajemy: wektor prawdziwych klas i wektor prawdopodobieństw klasy pozytywnej
#(zwrócony przez klasyfikator). Uwaga, jeżeli zmienna celu przyjmuje wartości różne niż {0,1} lub -1,1 musimy jawnie określić etykietę klasy pozytywnej za pomocą parametru `pos_label`. 
#Funkcja zwraca wektory zawierające wartości wskaźników *false positive rate* i *true positive rate* (czyli współrzędne punktów tworzących wykres krzywej ROC). 
#Dodatkowo otrzymujemy wektor zawierający progi decyzyjne, czyli wartości prawdopodobieństw wykorzystane do obliczenia współrzędnych punktów krzywej.
fpr, tpr, thresholds = roc_curve(y_test, probabilities[:,1],pos_label=1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#OX: false positive rate, czyli fpr=1-specyficznosc, OY: true positive rate, czyli tpr=czulosc
plt.plot(fpr, tpr, color='darkorange') 
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.show() 

#Pole pod krzywą ROC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, probabilities[:,1])







############################################################################

#Wczytujemy naszą nową bazę danych
churn_nowe=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab7\\churn_pl_dane_nowe.csv")

#Standaryzacja min-max
scaler_nowe=MinMaxScaler()
scaler_nowe.fit(churn_nowe[X_names_predyktory])
X_nowe=scaler_nowe.transform(churn_nowe[X_names_predyktory])

#Przewidywane wartości 
y_predicted_nowe=siec_neur.predict(X_nowe)  


##########################################################################



#Tworzymy nową listę predyktorów bez Nocminuty
#predyktory to"Czaswspolpracy","Liczbawiadomosci","Dzienminuty","Dzienrozmowy","Wieczorminuty","Wieczorrozmowy","Miedzynarodoweminuty","Miedzynarodowerozmowy","LiczbarozmowzBOK","Planmiedzy01"]

#Wykorzystujemy podział na zbiór uczący i testowy dokonany wczesniej. Ten wektor [0,1,2,3,4,5,8,9,10,11] to kolumny odpowiadające nszym predyktorom w X_train_norm, X_test_norm 
X2_train_norm=X_train_norm[:,[0,1,2,3,4,5,8,9,10,11]]
X2_test_norm=X_test_norm[:,[0,1,2,3,4,5,8,9,10,11]]

# 6 - to kolumna odpowiadająca zmiennej celu Nocminuty
y2_train_norm=X_train_norm[:,6]
y2_test_norm=X_test_norm[:,6]


#Budowa sieci 
from sklearn.neural_network import MLPRegressor
siec_neur2 = MLPRegressor(hidden_layer_sizes=(8,), activation='tanh',solver='lbfgs', alpha=0.0001, max_iter = 1000, random_state=123)
#Uczenie sieci
siec_train2=siec_neur2.fit(X2_train_norm,y2_train_norm)

#Przewidywane wartości 
y_predicted2=siec_neur2.predict(X2_test_norm) 

#Denormalizacja
import numpy as np
y_predicted_denor=np.zeros(y_predicted2.shape[0]) 

i=0
while i<= (y_predicted2.shape[0]-1):
      y_predicted_denor[i] =(y_predicted2[i]*(max(X_train['Nocminuty'])-min(X_train['Nocminuty'])))+min(X_train['Nocminuty'])
      
      i+=1

print(y_predicted_denor)



import math
from sklearn.metrics import mean_squared_error 
print('RMSE',math.sqrt(round(mean_squared_error(y_test2,y_predicted_denor),3))) 
