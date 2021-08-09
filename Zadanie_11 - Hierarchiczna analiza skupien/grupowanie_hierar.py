# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 13:54:12 2021

@author: user
"""

import pandas as pd
#Wczytujemy naszą bazę danych 
ceny=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\Lab11\\ceny.csv",sep=";",decimal=",")


#Standaryzacja zmiennych ilosciowych 
labels_ilosc = ['ryz', 'maka', 'kurczak', 'kielbasa', 'jaja','maslo','olej','cukier','miod','kawa','herbata','karp']
import numpy as np
from scipy import stats
ceny[labels_ilosc]=stats.zscore(ceny[labels_ilosc]) 

#Zestandaryzowane zmienne ilosciowe
data = ceny[labels_ilosc]


rok=['1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019']
# Hierarchiczna analiza skupien - podziel lata na grupy w zaleznosci od przecietnych cen zywnosci
#Rysowanie dendrogramu
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram = shc.dendrogram(shc.linkage(data, method='average',metric='euclidean'),labels=rok)


#Gdybymy użyli innej metody - single
plt.figure(figsize=(10, 7))
plt.title("Dendrogram - inna metoda")
dendrogram = shc.dendrogram(shc.linkage(data, method='single',metric='euclidean'),labels=rok)


#Wracamy do dendrogramu z metoda - average
#Algorytm - 2 klastry na podstawie dendogramu
from sklearn.cluster import AgglomerativeClustering
cluster1= AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average') #parametry algorytmu
cluster1.fit(data)  #stosujemy algorytm do naszych danych

#Zestawienie wygodne do opisu
ceny['klastry']=cluster1.labels_    #Atrybut labels_ przypisuje numery klastrów poszczególnym rekordom
zestawienie1=ceny[['rok','klastry','bezrobocie','przyrost2']]


# Hierarchiczna analiza skupien -  podziel zywnosc na grupy produktow, ktorych ceny zachowuja sie podobnie
data_transpozycja=data.T  # transpozycja macierzy ze standaryzowanymi zmiennymi ilosciowymi
data_transpozycja.columns=rok   #Dla estetyki nazywam kolumny odpowiednimi rocznikami

#Rysowanie dendrogramu
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram = shc.dendrogram(shc.linkage(data_transpozycja, metric='correlation', method='average'),labels=labels_ilosc)

#Algorytm - decydujemy sie na 3 klastry w oparciu o dendrogram
from sklearn.cluster import AgglomerativeClustering
cluster2 = AgglomerativeClustering(n_clusters=3, affinity='correlation', linkage='average') #parametry algorytmu
cluster2.fit(data_transpozycja)  #stosujemy algorytm do naszych danych
 
#Zestawienie wygodne do opisu
data_transpozycja['klastry']=cluster2.labels_     #Atrybut labels_ przypisuje numery klastrów poszczególnym rekordom
zestawienie2=data_transpozycja[['klastry']]


