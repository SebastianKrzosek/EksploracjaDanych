# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:23:19 2021

@author: user
"""

import pandas as pd
#Wczytujemy naszą bazę danych 
ceny=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab14\\ceny.csv",sep=";",decimal=",")

names=['ryz', 'maka', 'kurczak', 'kielbasa', 'jaja','maslo','olej','cukier','miod','kawa','herbata','karp']

#Macierz korelacji
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cm = np.corrcoef(ceny[names].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=names,xticklabels=names)
plt.show()

#Model regresji z wszystkimi predyktorami: "ryz", "maka", "kurczak", "kielbasa", "jaja","maslo","olej","cukier","miod","kawa","herbata"
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X=ceny[names]
y=ceny['bezrobocie']
model.fit(X,y)

#Przewidywane bezrobocie na podstawie cen produktów żywnosciowych w oparciu o zbudowany model regresji
y_pred=model.predict(X)
#Wartosci reszt
residuals = (y - y_pred)

#MSE i RMSE
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y, y_pred)
import math
RMSE=math.sqrt(MSE)
print(RMSE)

from scipy import stats
ceny[names]=stats.zscore(ceny[names]) 

from sklearn.decomposition import PCA
pca=PCA()  #ewentualnie: PCA(n_components=k), n_components-liczba komponentów, które mają zostać. Gdy nic nie wstawimy, to zostawiamy wszystkie. 
X_pca=pca.fit_transform(ceny[names])  #Skladowe głowne


import numpy as np
#Macierz transfomacji (macierz przejcia) przeksztalcajaca pierwotny zbior danych w zbior wyjsciowy uzyskany w wyniku analizy skladowych glownych  (wektory własne)
macierz=pca.components_.T
macierz_skladowych=pd.DataFrame(data=macierz,index=names,columns=['Skladowa 1','Skladowa 2','Skladowa 3','Skladowa 4','Skladowa 5','Skladowa 6','Skladowa 7','Skladowa 8','Skladowa 9','Skladowa 10','Skladowa 11','Skladowa 12'])

#Kryterium wartosci wlasnej
tabela=pd.DataFrame(data=pca.explained_variance_,index=['Skladowa 1','Skladowa 2','Skladowa 3','Skladowa 4','Skladowa 5','Skladowa 6','Skladowa 7','Skladowa 8','Skladowa 9','Skladowa 10','Skladowa 11','Skladowa 12'],columns=['Wartosc wlasna'])

#Kryterium czesci wariancji wyjasnianej przez skladowe glowne
calkowita_wyjasniona_wariancja=pd.DataFrame({"% wariancji":pca.explained_variance_ratio_,"% skumulowany":np.cumsum(pca.explained_variance_ratio_)},index=['Skladowa 1','Skladowa 2','Skladowa 3','Skladowa 4','Skladowa 5','Skladowa 6','Skladowa 7','Skladowa 8','Skladowa 9','Skladowa 10','Skladowa 11','Skladowa 12'])

#Kryterium wykresu osypiskowego
import matplotlib.pyplot as plt
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12],pca.explained_variance_,marker='o')
plt.axis([0, 12, -1, 11])
plt.title('Wykres osypiska')
plt.xlabel('Numer wartosci wlasnej')
plt.ylabel('wartosc wlasna')
plt.show()

#Wspolczynniki korelacji czastkowej
loadings=pca.components_.T * np.sqrt(pca.explained_variance_)
korelacja_czastkowa = pd.DataFrame(data=loadings,columns=['Skladowa 1','Skladowa 2','Skladowa 3','Skladowa 4','Skladowa 5','Skladowa 6','Skladowa 7','Skladowa 8','Skladowa 9','Skladowa 10','Skladowa 11','Skladowa 12'],index=names)

#Kryterium minimalnego zasobu zmiennoci wspólnej
dwie_skladowe=pd.DataFrame(data=korelacja_czastkowa['Skladowa 1']**2+korelacja_czastkowa['Skladowa 2']**2,columns=['Po wyodrebnieniu'])

#Model regresji w oparciu o skladowe glowne
model_new = LinearRegression()
X=X_pca[:,0:2]
y=ceny['bezrobocie']
model_new.fit(X,y)

#Przewidywane bezrobocie na podstawie składowych głóWnych w oparciu o zbudowany model regresji
y_pred_new=model_new.predict(X)
MSE_new=mean_squared_error(y, y_pred_new)
RMSE_new=math.sqrt(MSE_new)
print(RMSE_new)

#Moze przydatne zestawienie
zestawienie=pd.DataFrame({"bezrobocie realne":y,"%bezrobocie (12 zmiennych)":y_pred,"%bezrobocie (skladowe glowne)":y_pred_new})