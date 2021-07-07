# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:06:11 2020

@author: user
"""

import pandas as pd
import numpy as np

#Wczytujemy naszą bazę danych
churn=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab5\\churn_pl_dane.csv")

#Wyznaczamy macierz korelacji
korelacje=churn.corr()

#Predyktory jako X
X_names=["Czaswspółpracy","Liczbawiadomości","Dzieńminuty","Dzieńrozmowy","Wieczórminuty","Wieczórrozmowy","Nocminuty","Nocrozmowy","Międzynarodoweminuty","Międzynarodowerozmowy","LiczbarozmówzBOK","Planmiedzy01"]
X = churn[X_names]

#Zmienna celu jako y
y=churn[['Rezygnacja']]


#Dzielimy na zbiór uczący (70%) i zbiór testowy (30%)
#Dzięki parametrowi random_state określiliśmy ziarno losowości (random_state=123) dla wewnętrznego generatora liczb pseudolosowych. 
#Wprowadzenie ziarna o ustalonej wartości pozwala nam zachować odtwarzalność doświadczeń.
#Wykorzystujemy również wbudowaną obsługę nawarstwiania (stratyfikacji), przyjmującą postać
#wyrażenia stratify=y. W omawianym kontekście nawarstwianie oznacza, że metoda
#train_test_split zwraca podzbiory uczący i testowy mające takie same proporcje etykiet klas
#jak wejściowy zestaw danych uczących.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)


 #Polecenia `DecisionTreeClassifier` z pakietu `sklearn.tree`. Zgodnie z dokumentacją 
 #pakietu, drzewa budowane są za pomocą zoptymalizowanej wersji algorytmu *CART*, 
 #jednak aktualna implementacja nie wspiera zmiennych jakosciowych.
 #Najważniejsze parametry to:
 #- `criterion` - miara jakości podziału, możliwe wartości: `gini` (wartość domyślna) i `entropy`,
 #- `max_depth` - maksymalna głębokość drzewa, wartość domyślna `None`, czyli brak ograniczenia,
 #- `min_samples_split` - minimalna liczba elementów w węźle wymagana do jego podziału, domyślnie `2`. Zamiast liczby naturalnej można podać liczbę rzeczywistą, którą interpretujemy jako minimalny odsetek wszystkich elementów zbioru,
 #- `min_samples_leaf` - minimalna liczba elemntów w liściu, domyślnie `1`,
 #- `ccp_alpha` - współczynnik regularyzacji wykorzystywany w przycinaniu, domyślna wartość `0.0` (bez przycinania).
 #Więcej informacji można znaleźć w [dokumentacji](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) oraz w [podręczniku użytkownika](https://scikit-learn.org/stable/modules/tree.html).

#Tworzymy drzewo i trenujemy je na zbiorze uczącym
from sklearn.tree import DecisionTreeClassifier
# drzewo bez przyciecią ale z ograniczeniami
model1 = DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_split=10,min_samples_leaf=5) 
tree1train=model1.fit(X_train,y_train)

#Do narysowania drzewa wykorzystamy pakiet `graphviz`. 
#Uwaga zwykle trzeba go najpierw zainstalować (np. standardowa instalacja Anacondy go nie zawiera). 
#Możemy to zrobić wpisując w oknie poleceń Anacondy  `conda install python-graphviz`.
#Generowanie rysunku przebiega w dwóch etapach. Najpierw eksportujemy drzewo do formatu pośredniego za pomocą polecenia `export_graphviz`. 
#Informacje o tym poleceniu i jego parametrach można znaleźć w [dokumentacji](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz). Następnie przekazujemy te dane do polecenia `Source` z pakietu `graphviz`."

y_names=["Nie","Tak"]

#conda install python-graphviz 
#prawdopodobnie trzeba zainstalować ten pakiet

import graphviz 
from sklearn.tree import export_graphviz
wykres_drzewa1=export_graphviz(tree1train,out_file=None,filled=True,feature_names=X_names,class_names=y_names)
graph = graphviz.Source(wykres_drzewa1,format='png')
graph
#Pierwszy wpis w węźle określa warunek podziału. 
#Poniżej mamy wartość stosowanego kryterium czystości węzła, liczbę obserwacji wpadających do węzła oraz ich podział na klasy. 
#Kolory węzłów oznaczają klasę do któej należy większość obserwacji wpadających do danego węzła. 
#Im większa intensywność koloru, tym większa przewaga obserwacji z danej klasy.
graph.render("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab5\\decision_tree_graphivz1")



#Wyliczamy trafnosc na zbiorze uczacym i testowym
print('Trafnosc na zbiorze uczącym',round(model1.score(X_train,y_train),3))
print('Trafnosc na zbiorze testowym',round(model1.score(X_test,y_test),3)) 


#Przewidywane wartości i przewidywane prawdopodobieństwo
y_predicted1 = model1.predict(X_test)  
probabilities1=model1.predict_proba(X_test)

#Wyznaczmy macierz pomyłek i pozostałe miary jakości na zbiorze testowym
from sklearn.metrics import confusion_matrix
macierz_pomyłek=confusion_matrix(y_test,y_predicted1)
tn, fp, fn, tp = confusion_matrix(y_test,y_predicted1).ravel()
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
fpr, tpr, thresholds = roc_curve(y_test, probabilities1[:,1],pos_label=1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#OX: false positive rate, czyli fpr=1-specyficznosc, OY: true positive rate, czyli tpr=czulosc
plt.plot(fpr, tpr, color='darkorange') 
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.show() 

#Pole pod krzywą ROC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_predicted1)

#Wyznaczamy ważnosc predyktorow w modelu
model1.feature_importances_

#Rysujemy wykres ważnosci
waznosci = pd.Series(model1.feature_importances_, index=X_names)
waznosci.sort_values(inplace=True)
waznosci.plot(kind='barh')



#drzewo przycięte bo parametr ccp_alpha różny od 0 (daje się mały)
model2 = DecisionTreeClassifier(ccp_alpha=0.01,criterion="gini") 
tree2train=model2.fit(X_train,y_train)

wykres_drzewa2=export_graphviz(tree2train,out_file=None,filled=True,feature_names=X_names,class_names=y_names)
graph = graphviz.Source(wykres_drzewa2,format='png')
graph
graph.render("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab5\\decision_tree_graphivz2")


#Wyliczamy trafnosc na zbiorze uczacym i testowym
print('Trafnosc na zbiorze uczącym',round(model2.score(X_train,y_train),3))
print('Trafnosc na zbiorze testowym',round(model2.score(X_test,y_test),3)) 


#Przewidywane wartości i przewidywane prawdopodobieństwo
y_predicted2 = model2.predict(X_test)  
probabilities2=model2.predict_proba(X_test)

#Wyznaczmy macierz pomyłek i pozostałe miary jakości
from sklearn.metrics import confusion_matrix
macierz_pomyłek=confusion_matrix(y_test,y_predicted2)
tn, fp, fn, tp = confusion_matrix(y_test,y_predicted2).ravel()
print('Czułość: ', round(tp/(fn+tp),3))
print('Specyficzność: ', round(tn/(tn+fp),3))
print('Wskaźnik fałszywie negatywnych: ', round(fn/(fn+tp),3))
print('Wskaźnik fałszywie pozytywnych: ', round(fp/(tn+fp),3))
print('Precyzja: ', round(tp/(fp+tp),3))
print('Trafność: ', round((tp+tn)/(tp+fp+tn+fn),3))


#Tworzymy drzewo przycięte z równymi prawdopodobieństwami a priori i trenujemy je na zbiorze uczącym
model3 = DecisionTreeClassifier(ccp_alpha=0.005,criterion="gini",class_weight="balanced") #drzewo przycięte bo parametr ccp_alpha różny od 0 (daje się mały)
tree3train=model3.fit(X_train,y_train)

wykres_drzewa3=export_graphviz(tree3train,out_file=None,filled=True,feature_names=X_names,class_names=y_names)
graph = graphviz.Source(wykres_drzewa3,format='png')
graph
graph.render("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab5\\decision_tree_graphivz3")


#Wyliczamy trafnosc na zbiorze uczacym i testowym
print('Trafnosc na zbiorze uczącym',round(model3.score(X_train,y_train),3))
print('Trafnosc na zbiorze testowym',round(model3.score(X_test,y_test),3)) 


#Przewidywane wartości i przewidywane prawdopodobieństwo
y_predicted3 = model3.predict(X_test)  
probabilities3=model3.predict_proba(X_test)

#Wyznaczmy macierz pomyłek i pozostałe miary jakości
from sklearn.metrics import confusion_matrix
macierz_pomyłek=confusion_matrix(y_test,y_predicted3)
tn, fp, fn, tp = confusion_matrix(y_test,y_predicted3).ravel()
print('Czułość: ', round(tp/(fn+tp),3))
print('Specyficzność: ', round(tn/(tn+fp),3))
print('Wskaźnik fałszywie negatywnych: ', round(fn/(fn+tp),3))
print('Wskaźnik fałszywie pozytywnych: ', round(fp/(tn+fp),3))
print('Precyzja: ', round(tp/(fp+tp),3))
print('Trafność: ', round((tp+tn)/(tp+fp+tn+fn),3))

#Stosujemy ostatni model na nowych danych.
churn_nowe=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab5\\churn_pl_dane_nowe.csv")
churn_nowe['Rezygnacja']=model3.predict(churn_nowe[X_names])


#Tworzymy nową listę predyktorów bez Nocminuty
X_names2=["Czaswspółpracy","Liczbawiadomości","Dzieńminuty","Dzieńrozmowy","Wieczórminuty","Wieczórrozmowy","Nocrozmowy","Międzynarodoweminuty","Międzynarodowerozmowy","LiczbarozmówzBOK","Planmiedzy01"]
X2 = churn[X_names2]

#Zmienna celu Nocminuty jako y2
y2=churn[['Nocminuty']]

#Dzielimy na zbiór uczący (70%) i zbiór testowy (30%)
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=123)

#Tworzymy drzewo przewidujące ciagla zmienna celu i trenujemy je na zbiorze uczącym
#Dokumentacja na stronie 
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# drzewo bez przyciecią ale z ograniczeniami
model4 = DecisionTreeRegressor(criterion="mse",max_depth=5,min_samples_split=10,min_samples_leaf=5) 
tree4train=model4.fit(X2_train,y2_train)

wykres_drzewa4=export_graphviz(tree4train,out_file=None,filled=True,feature_names=X_names2)
graph = graphviz.Source(wykres_drzewa4,format='png')
graph
graph.render("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab5\\decision_tree_graphivz4")

#Wartosci przewidywane na zborach uczacym i testowym
y2pred_train = model4.predict(X2_train)
y2pred_test = model4.predict(X2_test)  

#Obliczamy MAE i pierwiastek z MSE na zbiorach uczacym i testowym

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math

print('MAE na zbiorze uczącym',round(mean_absolute_error(y2_train,y2pred_train),3))
print('MAE na zbiorze testowym',round(mean_absolute_error(y2_test,y2pred_test),3)) 

print('RMSE na zbiorze uczącym',round(math.sqrt(mean_squared_error(y2_train,y2pred_train)),3))
print('RMSE na zbiorze testowym',round(math.sqrt(mean_squared_error(y2_test,y2pred_test)),3)) 
