# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 13:54:12 2021

@author: user
"""

import pandas as pd
#Wczytujemy naszą bazę danych 
ceny=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab12\\ceny.csv",sep=";",decimal=",")

# Histogramy - czy nie ma problemów z długimi ogonami
import matplotlib.pyplot as plt
ceny['kawa'].plot(kind='hist',ylabel='Frequency') #histogram


#Standaryzacja zmiennych ilosciowych 
labels_ilosc = ['ryz', 'maka', 'kurczak', 'kielbasa', 'jaja','maslo','olej','cukier','miod','kawa','herbata','karp']
import numpy as np
from scipy import stats
ceny[labels_ilosc]=stats.zscore(np.sqrt(ceny[labels_ilosc]))


#Macierz korelacji
import seaborn as sns
import matplotlib.pyplot as plt
cm = np.corrcoef(ceny[labels_ilosc].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=labels_ilosc,xticklabels=labels_ilosc)
plt.show()


#Zestandaryzowane zmienne ilosciowe
data = ceny[labels_ilosc]


#Dwustopniowa analiza skupien do podziału lat na grupy zalezne od cen poszczegolnych produktow
from sklearn.cluster import Birch
brc = Birch(branching_factor=50, n_clusters=None, threshold=1.5) #the diameter of each leaf entry has to be less than threshold
brc.fit(data)
centroids=brc.subcluster_centers_  # Atrybut subcluster_centers wskazuje centra grup
ceny['klastry Birch']=brc.labels_         #Atrybut labels_ przypisuje numery klastrów poszczególnym rekordom - widzimy na ile grup podzielono  
     

#Rysowanie dendrogramu
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram = shc.dendrogram(shc.linkage(centroids,metric='euclidean', method='average'))

#Algorytm 
from sklearn.cluster import AgglomerativeClustering
cluster_new_Birch = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average') #parametry algorytmu
cluster_new_Birch.fit(centroids) #zapuszczamy na centrach grup z Birch
labels=cluster_new_Birch.labels_   


#Zestawienie wygodne do opisu            #Patrze jak laczone sa grupy z Bircha by otrzymac ostateczne klastry
ceny['klastry Birch ostatnie']=[1 if (item==0 or item==1) else 0 if (item==2 or item==3 or item==4) else 2 if (item==5) else item for item in ceny['klastry Birch']]
zestawienie3=ceny[['rok','klastry Birch','klastry Birch ostatnie','bezrobocie','przyrost2']]
 
#Miara sylwetki
from sklearn.metrics import silhouette_score
miara_sylwetki3=silhouette_score(centroids,cluster_new_Birch.labels_,metric='euclidean')
print('Silhouette Score(n=3):',miara_sylwetki3)






