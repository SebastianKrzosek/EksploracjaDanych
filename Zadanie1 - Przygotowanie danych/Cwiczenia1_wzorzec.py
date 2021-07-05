# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 00:28:22 2020

@author: JoannaKP
"""
# Importujemy pakiet pandas.
import pandas as pd


bank_train=pd.read_csv("C:\\Users\\user\\Documents\\Eksploracja danych\\Informatyka\\Lab1\\bank_marketing_training")

# Sprawdzamy ile rekordów i zmiennych ma zaimportowany plik.
bank_train.shape

# Dodajemy zmienną indeks. Pierwsza liczba to 0, bo Python numeruje od 0, 
# a ostatnia to liczba indeksow, ktore ma stworzyc.
bank_train['index']=pd.Series(range(0,26874))

# Wyswietlamy pierwszych i ostatnich 30 rekordow. Kolumna indeks jest na koncu!
bank_train.head 

# Rysujemy histogram zmiennej days_since_previous.
# Nad konsola klikamy Plots, by go obejrzec.
bank_train['days_since_previous'].plot(kind='hist', title='Histogram of Days Since Previous')

# Importujemy pakiet numpy.
import numpy as np

# Zamieniamy 999 na kod braku danych NaN.
bank_train['days_since_previous']=bank_train['days_since_previous'].replace({999: np.NaN})

# Rysujemy histogram zmiennej days_since_previous.
# Nad konsola klikamy Plots, by go obejrzec.
bank_train['days_since_previous'].plot(kind='hist', title='Histogram of Days Since Previous')

# Dupikujemy zmienną education i nazywamy ja education_numeric.
bank_train['education_numeric']=bank_train['education']

# Definiujemy słownik, który zastąpi kategorie zmiennej education_numeric liczbami.
# Słownik pojawił się w Variable explorer nad konsola.
dict_edu={"education_numeric": {"illiterate": 0, "basic.4y": 4, "basic.6y": 6, "basic.9y": 9,
                                "high.school": 12, "professional.course": 12, 
                                "university.degree": 16, "unknown": np.NaN}}
# Zamieniamy wartosci zmiennej na zdefiniowane w slowniku.
bank_train.replace(dict_edu, inplace=True)

# Importujemy pakiet stats.
from scipy import stats

# Standaryzujemy zmienna age.
bank_train['age_z']=stats.zscore(bank_train['age'])

# Identyfikujemy obserwacje odstajace o co najmniej 3 odchylenia standardowe.
bank_train.query('age_z>3 | age_z<-3')

# Tworzymy zmienna, ktora wskazuje obserwacje odstajace.
bank_train['age_outlier']=((bank_train['age_z']>3) | (bank_train['age_z']<-3))


