# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:04:03 2020

@author: user
"""

import pandas as pd


#Wczytujemy naszą bazę danych # do wczytywania plików excela:pd.ExcelFile, pd.read_excel
iris=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab2\\iris.csv")


#Macierz korelacji
cols = ['dlugosc_kielich', 'szerokosc_kielich', 'dlugosc_platek', 'szerokosc_platek', 'klasa']
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cm = np.corrcoef(iris[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.show()


#Macierz wykresów rozrzutów
import matplotlib.pyplot as plt
import seaborn as sns
names = ["dlugosc_kielich", "szerokosc_kielich", "dlugosc_platek", "szerokosc_platek"]
sns.pairplot(iris,x_vars=names,y_vars=["klasa"],diag_kind = None)
plt.tight_layout()
plt.show()




#Trójwymiarowy wykres rozrzutu: zależność ceny karpia od jej dwóch najważniejszych predyktorów 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x=np.array(iris['dlugosc_kielich'])
y=np.array(iris['dlugosc_platek'])
z=np.array(iris['klasa'])
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('Dlugosc kielicha')
ax.set_ylabel('Dlugosc platek')
ax.set_zlabel('Gatunek')
plt.show()