# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:19:05 2020

@author: JoannaKP
"""

# Importujemy pakiet pandas, numpy i pyplot z matplotlib.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Otwieramy dane ustawiajac sciezke do odpowiedniego katalogu. 
bank_train=pd.read_csv("C:\\User\\Eksploracja danych\\bank_marketing_training")

# Będziemy rysować wykres słupkowy zestawiony - zwykły i znormalizowany.
# Tworzymy tabelę krzyżową zestawiającą wartosci zmiennych previous_outcome i response.
crosstab_por=pd.crosstab(bank_train['previous_outcome'],bank_train['response'])

# W oparciu o tabelę tworzymy wykres słupkowy (bar chart) zestawiony (stacked).
crosstab_por.plot(kind='bar', stacked=True)

# Zmieniamy kolor słupków, kolejnosc (zeby było yes na dole a no u gory) i kolejnosc legendy na taka sama jak wykresu.
wykres_por = crosstab_por[["yes","no"]].plot(kind='bar', stacked=True, color=['green', 'red'])
handles, labels = wykres_por.get_legend_handles_labels()
wykres_por.legend(reversed(handles), reversed(labels))

# Tworzymy nową tabelę z odsetkami zamiast liczebnosci, odsetki liczone sa w wierszach (normalize='index'). 
# Gdyby miały być w kolumnach, ustawiamy normalize='columns'.
crosstab_por_norm=pd.crosstab(bank_train['previous_outcome'],bank_train['response'], normalize='index')

# Wykonujemy wykres słupkowy znormalizowany.
wykres_por_norm = crosstab_por_norm[["yes","no"]].plot(kind='bar', stacked=True, color=['green', 'red'])
handles, labels = wykres_por_norm.get_legend_handles_labels()
wykres_por_norm.legend(reversed(handles), reversed(labels))

# Bedziemy teraz tworzyc tabele krzyzowe 
#Wykonujemy tabelę krzyżową zestawiajac wartosci zmiennej response (w wierszach) i previous_outcome (w kolumnach).
crosstab_rpo=pd.crosstab(bank_train['response'], bank_train['previous_outcome'])

# Wykonujemy tabelę z zaokraglonymi do 1 miejsca po przecinku procentami w kolumnach.
crosstab_rpo_proc=round(pd.crosstab(bank_train['response'], bank_train['previous_outcome'], normalize='columns')*100,1)

# Bedziemy teraz tworzyc histogramy i histogramy znormalizowane


# Separujemy wartosci zmiennej age dla rekordow z wartosciami yes i no zmiennej response.
bank_train_age_resp_y=bank_train[bank_train.response=="yes"]['age']
bank_train_age_resp_n=bank_train[bank_train.response=="no"]['age']

# Wykonujemy histogram (trzeba zaznaczyć i uruchomić całosc)
plt.hist([bank_train_age_resp_y, bank_train_age_resp_n], bins=10, stacked=True, color=['green', 'red'], label = ['Response=Yes', 'Response=No' ])
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(reversed(handles), reversed(labels))
plt.title('Histogram of Age with Response Overlay')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Żeby wykonać histogram znormalizowany sczytujemy pewne ustawienia z tego, ktory wykonalismy.
# n - wysokosci slupkow dla yes i no, bins - granice kazdego ze slupkow
(n, bins, patches)=plt.hist([bank_train_age_resp_y, bank_train_age_resp_n], bins=10, stacked=True)

# W oparciu o n obliczamy wartosci procentowe potrzebne do wykonania znormalizowanego histogramu.
# Budujemy tabelę.
n_table=np.column_stack((n[0], n[1]))

# Obliczamy odsetki.
n_norm=n_table/n_table.sum(axis=1)[:,None]

# Tworzymy tabelę z początkami i koncami slupkow histogramu.
ourbins=np.column_stack((bins[0:10], bins[1:11]))

# Tworzymy znormalizowany histogram.
p1=plt.bar(x=ourbins[:,0], height=n_norm[:,0], width=ourbins[:,1] - ourbins[:,0], color='green', label = 'Response=Yes')
p2=plt.bar(x=ourbins[:,0], height=n_norm[:,1], width=ourbins[:,1] - ourbins[:,0], bottom=n_norm[:,0], color='red', label = 'Response=No')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(reversed(handles), reversed(labels))
plt.title('Normalized Histogram of Age with Response Overlay')
plt.xlabel('Age');
plt.ylabel('Proportion')
plt.show()

# Dzielimy zmienną age na kategorie wiekowe. Musimy dać 60,01, żeby przedział zawierał 60.
bank_train['age_binned'] = pd.cut(x=bank_train['age'], bins=[0,27,60.01,100], labels=["Under 27", "27 to 60", "Over 60"], right=False)

# Tworzymy tabelę krzyżową zestawiającą wartosci zmiennych age_binned i response.
crosstab_abr=pd.crosstab(bank_train['age_binned'],bank_train['response'])

# Wykonujemy zestawiony wykres słupkowy.
wykres_abr=crosstab_abr[["yes","no"]].plot(kind='bar', stacked=True, color=['green', 'red'])
handles, labels = wykres_abr.get_legend_handles_labels()
wykres_abr.legend(reversed(handles), reversed(labels))

# Tworzymy nową tabelę z odsetkami zamiast liczebnosci, odsetki liczone sa w wierszach.
crosstab_abr_norm=pd.crosstab(bank_train['age_binned'],bank_train['response'], normalize='index')

# Wykonujemy wykres słupkowy znormalizowany.
wykres_abr_norm=crosstab_abr_norm[["yes","no"]].plot(kind='bar', stacked=True, color=['green', 'red'])
handles, labels = wykres_abr_norm.get_legend_handles_labels()
wykres_abr_norm.legend(reversed(handles), reversed(labels))