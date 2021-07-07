# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from sklearn import neighbors, datasets
#Pod nazwą iris zapisujemy naszą bazę danych.
iris = datasets.load_iris()

# Przydatne komendy z pakietu scikit-learn
#print(iris.DESCR) # opis danych
# rozmiary tablicy celu # features, cechy
#print(type(iris.data)) # tablica NumPy ndarray
#print(iris.data.shape) # rozmiar tablicy zwraca tuple
#print(iris.feature_names) # lista nazw dla iris.data
#print(iris.target)  # zmienna celu (target)
#print(iris.target.shape) #rozmiary tablicy celu



#Cztery pierwsze zmienne (predyktory) wykorzystane do nauki problemu

#Wypiszmy ich nazwy
print(iris.feature_names)

#Wczytujemy je wszystkie jako X
X = iris.data[:, :4]

#Wczytujemy zmienna celu
y = iris.target
print('Etykiety klas dla zmiennej celu:', np.unique(y)) #nazwy trzech gatunków (Iris setosa, Iris
#versicolor oraz Iris virginica) są już przechowywane w postaci liczb całkowitych (tutaj 0, 1, 2).



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5, stratify=y)
#Zwróć uwagę, że funkcja train_test_split przed rozdzieleniem podzbiorów przeprowadza
#wewnętrzne tasowanie zestawów uczących; gdyby nie to, wszystkie przykłady z klas 0 i 1
#znalazłyby się w zbiorze uczącym, a zbiór testowy składałby się z 45 przykładów należących
#do klasy 2. Dzięki parametrowi random_state określiliśmy ziarno losowości (random_state=5)
#dla wewnętrznego generatora liczb pseudolosowych, który służy do tasowania zestawów danych
#przed ich rozdzieleniem. Wprowadzenie ziarna o ustalonej wartości pozwala nam zachować
#odtwarzalność doświadczeń.
#Wykorzystujemy również wbudowaną obsługę nawarstwiania (stratyfikacji), przyjmującą postać
#wyrażenia stratify=y. W omawianym kontekście nawarstwianie oznacza, że metoda
#train_test_split zwraca podzbiory uczący i testowy mające takie same proporcje etykiet klas
#jak wejściowy zestaw danych uczących. Możemy użyć funkcji bincount (biblioteka NumPy)
#zliczającej wystąpienia każdej wartości w danej tablicy i przekonać się, że rzeczywiście mamy
#do czynienia z nawarstwianiem:

print('Liczba etykiet w zbiorze y:', np.bincount(y))
print('Liczba etykiet w zbiorze y_train:', np.bincount(y_train))
print('Liczba etykiet w zbiorze y_test:', np.bincount(y_test))

from scipy import stats
#Standaryzujemy predyktory w próbie uczącej i próbie testowej
X_train = stats.zscore(X_train)
X_test = stats.zscore(X_test)


from sklearn.neighbors import KNeighborsClassifier
#Algorytm k_NN dla: k=3, bez ważenia, z metryką euklidesową.
knn = KNeighborsClassifier(3, weights='uniform', metric='euclidean')
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
probabilities=knn.predict_proba(X_test)

#Tworzymy macierz pomyłek
from sklearn.metrics import confusion_matrix
macierz_pomyłek = confusion_matrix(y_test, y_pred)



#Definiujemy pomocniczo wektory zbudowane z samych zer długosci odpowiednio rozmiarowi y_test (to samo co y_pred)
y_test_setosa=np.zeros(y_test.data.shape[0])
y_pred_setosa=np.zeros(y_pred.data.shape[0])

#setosa - kodujemy setosa w y_test i y_pred jako 1 (bo krzywa ROC w odniesieniu do setosa), a pozostałe łącznie jako 0.

licznik=0
while licznik<= (y_test.data.shape[0]-1):#długoćć wektora to 45, ale numeracja jest od 0 stąd trzeba odjąć 1.
    if y_test[licznik]==0:
        y_test_setosa[licznik]=1
    else:
        y_test_setosa[licznik]=0 
         
    licznik += 1 
        
licznik=0    
while licznik<= (y_pred.data.shape[0]-1):
    if y_pred[licznik]==0:
        y_pred_setosa[licznik]=1
    else:
        y_pred_setosa[licznik]=0 
        
    licznik += 1
    
#Krzywa ROC dla setosa    
from sklearn import metrics
#import numpy as n                Jest potrzebny, ale wywołalimy go na początku
import matplotlib.pyplot as plt
probs_setosa=probabilities[:,0]
#Funkcja `roc_curve` oblicza wartości potrzebne do narysowania wykresu (ale nie rysuje wykresu). 
#Funkcja ta działa tylko w przypadku klasyfikacji binarnej. Jako parametry podajemy: wektor prawdziwych klas i wektor prawdopodobieństw klasy pozytywnej
#(zwrócony przez klasyfikator). Uwaga, jeżeli zmienna celu przyjmuje wartości różne niż {0,1} lub -1,1 musimy jawnie określić etykietę klasy pozytywnej za pomocą parametru `pos_label`. 
#Funkcja zwraca wektory zawierające wartości wskaźników *false positive rate* i *true positive rate* (czyli współrzędne punktów tworzących wykres krzywej ROC). 
#Dodatkowo otrzymujemy wektor zawierający progi decyzyjne, czyli wartości prawdopodobieństw wykorzystane do obliczenia współrzędnych punktów krzywej.
fpr, tpr, thresholds = metrics.roc_curve(y_test_setosa, probs_setosa,pos_label=1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#OX: false positive rate, czyli fpr=1-specyficznosc, OY: true positive rate, czyli tpr=czulosc
plt.plot(fpr, tpr, color='darkorange') 
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.show() 



#Definiujemy pomocniczo wektory zbudowane z samych zer długosci odpowiednio rozmiarowi y_test (to samo co y_pred)
y_test_versicolor=np.zeros(y_test.data.shape[0])
y_pred_versicolor=np.zeros(y_pred.data.shape[0])
#versicolor - kodujemy versicolor w y_test i y_pred jako 1 (bo krzywa ROC w odniesieniu do versicolor), a pozostałe łącznie jako 0.
#Ponieważ versicolor był kodowany jako 1, to pozostałe kodujemy jako 0.
licznik=0
while licznik<= (y_test.data.shape[0]-1):
    if y_test[licznik]!=1:
        y_test_versicolor[licznik]=0
    else:
        y_test_versicolor[licznik]=1

    licznik += 1 
        
licznik=0   
while licznik<= (y_pred.data.shape[0]-1):
    if y_pred[licznik]!=1:
        y_pred_versicolor[licznik]=0
    else:
        y_test_versicolor[licznik]=1
     
    licznik += 1
    
#Krzywa ROC dla versicolor   
from sklearn import metrics
#import numpy as np          Jest potrzebny, ale wywołalimy go na początku
import matplotlib.pyplot as plt
probs_versicolor=probabilities[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test_versicolor, probs_versicolor,pos_label=1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#OX: false positive rate, czyli fpr=1-specyficznosc, OY: true positive rate, czyli tpr=czulosc
plt.plot(fpr, tpr, color='darkorange') 
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.show() 
 

#Definiujemy pomocniczo wektory zbudowane z samych zer długosci odpowiednio rozmiarowi y_test (to samo co y_pred) 
y_test_virginica=np.zeros(y_test.data.shape[0])
y_pred_virginica=np.zeros(y_pred.data.shape[0]) 
#virginica - kodujemy virginica w y_test i y_pred jako 1 (bo krzywa ROC w odniesieniu do versicolor), a pozostałe łącznie jako 0.
licznik=0
while licznik<= (y_test.data.shape[0]-1):
    if y_test[licznik]==2:
        y_test_virginica[licznik]=1
    else:
        y_test_virginica[licznik]=0
        
    licznik += 1 
        

licznik=0
while licznik<= (y_pred.data.shape[0]-1):
    if y_pred[licznik]==2:
        y_pred_virginica[licznik]=1
    else:
        y_pred_virginica[licznik]=0
        
    licznik += 1
    
 #Krzywa ROC dla virginica   
from sklearn import metrics
#import numpy as np               Jest potrzebny, ale wywołalimy go na początku
import matplotlib.pyplot as plt
probs_virginica=probabilities[:,2]
fpr, tpr, thresholds = metrics.roc_curve(y_test_virginica, probs_virginica, pos_label=1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#OX: false positive rate, czyli fpr=1-specyficznosc, OY: true positive rate, czyli tpr=czulosc
plt.plot(fpr, tpr, color='darkorange') 
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.show() 
