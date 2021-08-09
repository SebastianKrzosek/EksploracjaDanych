# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 19:42:02 2021

@author: user
"""

import pandas as pd
#Wczytujemy naszą bazę danych stragan 
stragan=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Laboratorium\\Reguly_asocjacyjne_lab14\\stragan.csv")

names=['Szparagi', 'Fasola', 'Brokuły', 'Kukurydza','Zielona_papryka', 'Kabaczki', 'Pomidory']

#Koniecznosc rekodowania wartosci na 0 i 1 lub False i True. My rekodujemy na 1 i 0.
dict_Szparagi={"Szparagi": {"yes": 1, "no": 0}}
stragan.replace(dict_Szparagi, inplace=True)

dict_Fasola={"Fasola": {"yes": 1, "no": 0}}
stragan.replace(dict_Fasola, inplace=True)

dict_Brokuły={"Brokuły": {"yes": 1, "no": 0}}
stragan.replace(dict_Brokuły, inplace=True)

dict_Kukurydza={"Kukurydza": {"yes": 1, "no": 0}}
stragan.replace(dict_Kukurydza, inplace=True)

dict_Zielona_papryka={"Zielona_papryka": {"yes": 1, "no": 0}}
stragan.replace(dict_Zielona_papryka, inplace=True)

dict_Kabaczki={"Kabaczki": {"yes": 1, "no": 0}}
stragan.replace(dict_Kabaczki, inplace=True)

dict_Pomidory={"Pomidory": {"yes": 1, "no": 0}}
stragan.replace(dict_Pomidory, inplace=True)


#Biblioteka mlxtend nie dostepna standardowo. Trzeba zainstalowac z zewnatrz.
#pip install mlxtend  

from mlxtend.frequent_patterns import apriori
#Minimalne wsparcie 0.2
reguly_stragan1 = apriori(stragan[names], min_support=0.2,use_colnames=True)

from mlxtend.frequent_patterns import association_rules
#Dodatkowo minimalna ufnosc 0.75
reguly_stragan2=association_rules(reguly_stragan1, metric="confidence", min_threshold=0.75)



#Wczytujemy naszą bazę danych ubrania
ubrania=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Laboratorium\\Reguly_asocjacyjne_lab14\\ubrania.csv",sep=";",decimal=",")

#Wsparcie minimalne 0.2
rules1 = apriori(ubrania, min_support=0.2,use_colnames=True)
#Dodatkowo ufnosc minimalna 0.75
rules1a=association_rules(rules1, metric="confidence", min_threshold=0.75)

#Wsparcie minimalne 0.3
rules2 = apriori(ubrania, min_support=0.3,use_colnames=True)
#Dodatkowo ufnosc minimalna 0.75
rules2b=association_rules(rules2, metric="confidence", min_threshold=0.75)

#Wsparcie minimalne 0.2
rules3 = apriori(ubrania, min_support=0.2,use_colnames=True)
#Dodatkowo ufnosc minimalna 0.9
rules3c=association_rules(rules3, metric="confidence", min_threshold=0.9)

#Wsparcie minimalne 0.3
rules4 = apriori(ubrania, min_support=0.3,use_colnames=True)
#Dodatkowo ufnosc minimalna 0.75
rules4d=association_rules(rules4, metric="confidence", min_threshold=0.9)






