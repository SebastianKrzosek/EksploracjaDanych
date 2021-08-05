# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:04:03 2020

@author: user
"""

import pandas as pd


#Wczytujemy naszą bazę danych # do wczytywania plików excela:pd.ExcelFile, pd.read_excel
ceny=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab9\\karp.csv")


#Macierz korelacji
cols = ['ryz', 'maka', 'kurczak', 'kielbasa', 'jaja','maslo','olej','cukier','miod','kawa','herbata','karp']
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cm = np.corrcoef(ceny[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.show()


#Macierz wykresów rozrzutów
import matplotlib.pyplot as plt
import seaborn as sns
names = ["ryz", "maka", "kurczak", "kielbasa", "jaja","maslo","olej","cukier","miod","kawa","herbata"]
sns.pairplot(ceny,x_vars=names,y_vars=["karp"],diag_kind = None)
plt.tight_layout()
plt.show()



#Model regresji z wszystkimi predyktorami: "ryz", "maka", "kurczak", "kielbasa", "jaja","maslo","olej","cukier","miod","kawa","herbata"
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X=ceny[names]
y=ceny['karp']
model.fit(X,y)
# Wspolczynniki rowanania regresji odpowiednio przy kolejnych predyktorach
model.coef_
# Wyraz wolny w rownaniu regresji
model.intercept_


#Selekcja predyktorow w oparciu o RFE
#dokumentacja: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
from sklearn.feature_selection import RFE
#n_features_to_select=2, wybieram dwa najważniejsze predyktory
selector1 = RFE(model, n_features_to_select=2, step=1)
selector1 = selector1.fit(X, y)
#True wskazuje ktore dwa sposrod zaproponowanych ("ryz", "maka", "kurczak", "kielbasa", "jaja","maslo","olej","cukier","miod","kawa","herbata") sa wybierane
selector1.support_
#Ranking dolaczania poszczegolnych predyktorow
selector1.ranking_


#Model regresji z dwoma najwazniejszymi predyktorami wybranymi w powyzszej selekcji, czyli "olej" i "jaja"
model_new=LinearRegression()
names_new=["jaja","olej"]
X_new=ceny[names_new]   # y, czyli cena karpia bez zmian
model_new.fit(X_new,y)

#Wpsolczynnik R^2 mozna tez policzyc tak:
model_new.score(X_new,y)

# Wspolczynniki rowaniania regresji odpowiednio przy 'jaja' i 'olej'
model_new.coef_
# Wyraz wolny w rownaniu regresji
model_new.intercept_

#Przewidywana cena karpia na podstawie ceny oleju i jaj w oparciu o zbudowany model regresji
y_pred=model_new.predict(X_new)
#Wartosci reszt
residuals = (y - y_pred)


#Wykres Q-Q dla reszt
from scipy import stats
import matplotlib.pyplot as plt
stats.probplot(residuals,dist="norm",plot=plt)

#Test normalnosci dla reszt, jesli pvalue>0.05, to mozna przyjac, ze maja rozklad normalny
from scipy import stats
shapiro_test = stats.shapiro(residuals)
print(shapiro_test)


#Test niezaleznosci dla reszt. Im blizsza 2 wartosc otrzymanej statystyki tym bardziej mozna przyjac niezaleznosc reszt 
# Wiecej o tescie Durbina-Watsona w Pythonie: https://www.statology.org/durbin-watson-test-python/
from statsmodels.stats.stattools import durbin_watson 
d_b = durbin_watson(residuals) 
print(d_b)


#Blad sredniokwadratowy i jego pierwiastek
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y, y_pred)
import math
S=math.sqrt(MSE)
print('standardowy bład oszacowania', S)


#Standaryzujemy reszty w oparciu o wzor z wykladu
import statistics #by liczyc mean
i=0
suma=0
N=residuals.shape[0] #liczba wyznaczonych reszt=liczba obserwacji
while i<=(N-1):
    suma=suma+(y[i]-statistics.mean(y))**2
    i+=1

i=0 
h=np.zeros(N)
while i<=(N-1):
    h[i]=1/N+(((y[i]-statistics.mean(y))**2)/suma)
    i+=1

i=0
residuals_stand=np.zeros(N)
while i<=(N-1):
    residuals_stand=residuals/(S*math.sqrt(1-h[i]))
    i+=1
print('Standaryzowane reszty', residuals_stand)


#Standaryzujemy wartosci przewidywane ceny karpia
# Importujemy pakiet stats.
#from scipy import stats  - my to zrobilismy wczesniej
y_pred_stand=stats.zscore(y_pred)
print('Standaryzowane wartosci przewidywane', y_pred_stand)


# Wykres standaryzowanych reszt wzgledem standaryzowanych wartosci przewidywanych
import matplotlib.pyplot as plt
plt.scatter(y_pred_stand,residuals_stand)
plt.title('Zaleznosc standaryzowanych reszt wzgledem standaryzowanych wartosci przewidywanych')
plt.xlabel('Stand. przewidywana cena karpia')
plt.ylabel('Stand. reszty')
plt.show()



#Trójwymiarowy wykres rozrzutu: zależność ceny karpia od jej dwóch najważniejszych predyktorów (u nas cena jaja i cena oleju)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x=np.array(ceny['jaja'])
y=np.array(ceny['olej'])
z=np.array(ceny['karp'])
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('Cena jaj')
ax.set_ylabel('Cena oleju')
ax.set_zlabel('Cena karpia')
plt.show()