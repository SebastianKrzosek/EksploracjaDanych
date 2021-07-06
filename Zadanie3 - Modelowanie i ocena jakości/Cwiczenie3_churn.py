# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:04:03 2020

@author: user
"""

import pandas as pd


#Wczytujemy naszą bazę danych # do wczytywania plików excela:pd.ExcelFile, pd.read_excel
churn=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab3\\churn_pl_klasyfikacja.csv")

#Próba ucząca
rezygnacja_train=churn['Rezygnacja'][0:2298]
rezygnacja_predict_train=churn['Rezygnacja_przewidywane'][0:2298]

#Tworzymy macierz pomyłek dla próby uczącej
from sklearn.metrics import confusion_matrix
macierz_pomyłek1 = confusion_matrix(rezygnacja_train,rezygnacja_predict_train)

#Próba testowa
rezygnacja_test=churn['Rezygnacja'][2298:3300]
rezygnacja_predict_test=churn['Rezygnacja_przewidywane'][2298:3300]

#Tworzymy macierz pomyłek dla próby testowej
macierz_pomyłek2 = confusion_matrix(rezygnacja_test,rezygnacja_predict_test)





 #Krzywa ROC dla próby uczącej  
from sklearn import metrics            
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(rezygnacja_train, churn['Prawdopodobieństwo_1'][0:2298], pos_label=1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#OX: false positive rate, czyli fpr=1-specyficznosc, OY: true positive rate, czyli tpr=czulosc
plt.plot(fpr, tpr, color='darkorange') 
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.show() 

#Pole pod krzywą ROC
from sklearn.metrics import roc_auc_score
roc_auc_score(rezygnacja_train, churn['Prawdopodobieństwo_1'][0:2298])


 #Krzywa ROC dla próby testowej 
from sklearn import metrics
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(rezygnacja_test,churn['Prawdopodobieństwo_1'][2298:3300], pos_label=1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#OX: false positive rate, czyli fpr=1-specyficznosc, OY: true positive rate, czyli tpr=czulosc
plt.plot(fpr, tpr, color='darkorange') 
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.show() 

#Pole pod krzywą ROC
roc_auc_score(rezygnacja_test, churn['Prawdopodobieństwo_1'][2298:3300])