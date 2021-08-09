w# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 16:44:03 2021

@author: user
"""

import pandas as pd

#Wczytujemy naszą bazę danych 
telco=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab10\\dane_telco.csv", sep=";",decimal=",")


#Macierz korelacji
cols = ['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon','longten','tollten', 'equipten', 'cardten', 'wireten']
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cm = np.corrcoef(telco[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.show()



# Histogramy i wykresy skrzynkowe (zmienne ilosciowe)
ax = sns.boxplot(data=telco['longmon']) #wykres skrzynkowy
#ax_razem = sns.boxplot(data=telco[labels_ilosc], orient="h", color="gray")
telco['longmon'].plot(kind='hist', title='Histogram of Longmon') #histogram
 

Predyktory_names=['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon','multline','voice','pager','internet','callid','callwait','forward','confer','ebill']
labels_ilosc=['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon']
labels_jak=['multline','voice','pager','internet','callid','callwait','forward','confer','ebill'] 

#Operacje na zmiennych ilosciowych i jakosciowych
from scipy import stats
telco[labels_ilosc]=stats.zscore(np.log(telco[labels_ilosc]+1)) #zmienne ilosciowe
telco[labels_jak]=stats.zscore(telco[labels_jak])               #zmienne jakosciowe
Predyktory=telco[Predyktory_names]

#Histogram po przeksztalceniach
Predyktory['longmon'].plot(kind='hist', title='Histogram of Longmon') #histogram

#Trójwymiarowy wykres rozrzutu
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x=np.array(Predyktory['longmon'])
y=np.array(Predyktory['equipmon'])
z=np.array(Predyktory['tollmon'])
ax.scatter(x, y, z, c='r', marker='o')
plt.show()


#Algorytm k-srednich, ustalajac najpierw k=3. 
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
#km = KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,tol=1e-04,random_state=0)
kmeans=km.fit(Predyktory)
cluster=kmeans.labels_
Cluster0=Predyktory.loc[cluster==0]
Cluster1=Predyktory.loc[cluster==1]
Cluster2=Predyktory.loc[cluster==2]
opis1=Cluster0.describe()
opis2=Cluster1.describe()
opis3=Cluster2.describe()




#Algorytm k-srednich, ustalajac teraz k=4. 
km_new = KMeans(n_clusters=4,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
#km = KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,tol=1e-04,random_state=0)
kmeans_new=km_new.fit(Predyktory)
cluster_new=kmeans_new.labels_
Cluster0_new=Predyktory.loc[cluster_new==0]
Cluster1_new=Predyktory.loc[cluster_new==1]
Cluster2_new=Predyktory.loc[cluster_new==2]
Cluster3_new=Predyktory.loc[cluster_new==3]
opis1_new=Cluster0_new.describe()
opis2_new=Cluster1_new.describe()
opis3_new=Cluster2_new.describe()
opis4_new=Cluster3_new.describe()


#Rysujemy histogramy i histogramy znormalizowane dla klastrow/grup. Robimy to dla grupy 0 (Cluster0). Pozostałe podobnie.

#Wybieramy te obserwacje zmiennej churn, ktore sa zwiazane z klastrem 0 (grupa 0)
churn_0=telco['churn'][cluster_new==0]
# Separujemy wartosci zmiennej churn ze wzgledu na yes i no 
churn_0_no=churn_0[churn_0==0]
churn_0_yes=churn_0[churn_0==1]

# Wykonujemy histogram (trzeba zaznaczyć i uruchomić całosc)
plt.hist([churn_0_yes,churn_0_no], bins=1, stacked=True, color=['green', 'red'], label = ['Response=Yes', 'Response=No' ])
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(reversed(handles), reversed(labels))
plt.title('Histogram of Grupa 0')
plt.xlabel('Churn')
plt.ylabel('Frequency')
plt.show()


# Żeby wykonać histogram znormalizowany szczytujemy pewne ustawienia z tego, ktory wykonalismy.
# n - wysokosci slupkow dla yes i no, bins - granice kazdego ze slupkow
(n, bins, patches)=plt.hist([churn_0_yes,churn_0_no], bins=1, stacked=True)

# W oparciu o n obliczamy wartosci procentowe potrzebne do wykonania znormalizowanego histogramu.
# Budujemy tabelę.
n_table=np.column_stack((n[0], n[1]))

# Obliczamy odsetki.
n_norm=n_table/n_table.sum(axis=1)[:,None]

# Tworzymy tabelę z początkami i koncami slupkow histogramu.
ourbins=np.column_stack((bins[0:1], bins[1:2]))

# Tworzymy znormalizowany histogram.
p1=plt.bar(x=ourbins[:,0], height=n_norm[:,0], width=ourbins[:,1] - ourbins[:,0], color='green', label = 'Response=Yes')
p2=plt.bar(x=ourbins[:,0], height=n_norm[:,1], width=ourbins[:,1] - ourbins[:,0], bottom=n_norm[:,0], color='red', label = 'Response=No')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(reversed(handles), reversed(labels))
plt.title('Normalized Histogram of Grupa 0')
plt.xlabel('Churn');
plt.ylabel('Proportion')
plt.show()

