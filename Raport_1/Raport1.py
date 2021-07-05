# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import sys
from plotnine import ggplot, aes, geom_line, geom_bar, coord_flip, labs, scale_fill_discrete
import sys
import seaborn as sns
from scipy import stats


# %%
#zad1 - wczytanie pliku csv
cereals=pd.read_csv("./cereals.CSV")


# %%
#zad2.1 - stworzenie zwykłego wykresu słupkowego
pd.crosstab(cereals.Manuf,cereals.Shelf).plot.bar(stacked=True)


# %%
#zad2.2 - stworzenie znormalizowanego wykresu słupkowego
pd.crosstab(cereals.Manuf,cereals.Shelf, normalize="index").plot.bar(stacked=True)


# %%
# zad3 - stworzenie tabeli krzyrzowej zmiennej Manuf w kolumnach i Shelf w wierach. Dane przedstawione w formie procentowej
pd.crosstab(index = cereals["Shelf"], columns =  cereals["Manuf"], normalize=True).round(4)*100


# %%
# zad4 - wyliczenie obserwacji odstajacych. Standaryzacja wykonana na podstawie wzoru, a nastepnie sprawdzenie czy wpada w przedzial [-3, 3]
cereals["Potass_z"]=(cereals["Potass"]-cereals["Potass"].mean())/cereals["Potass"].std()
print(cereals.query('Potass_z>3 | Potass_z<-3')["Name"])
cereals['Potass_outlier']=((cereals['Potass_z']>3) | (cereals['Potass_z']<-3))
print("")
print("Obserwacji odstajacych: ", end='')
sum(cereals["Potass_outlier"]==True)


# %%
# zad5.1 - zwykły histogram zmiennej Potass z oznaczeniem kolorem Shelf'a
sns.histplot(cereals, x="Potass", hue="Shelf", shrink=.9, multiple = "stack", palette=sns.color_palette("husl", 3))


# %%
# zad5.2 - znormalizowany histogram zmiennej Potass z oznaczeniem kolorem Shelf'a
sns.histplot(cereals, x="Potass", hue="Shelf", multiple="fill", palette=sns.color_palette("husl", 3), shrink = .9)


# %%
# zad6.1 - Stworzenie zmiennej Calories_binned na podstawie podanych przedzialów
cut_bins = [0, 89.99, 110, 200] # przedziały zostały przygotowane zgodnie z instrukcjami podanymi na zajeciach 
cereals['Calories_binned'] = pd.cut(cereals['Calories'], bins=cut_bins, labels=["0 < Calories < 90","90 <= x <= 110","x >= 110"])


# %%
# zad 6.2 - Stworzenie zwykłego wykresu słupkowego dla nowej zmiennej zaznaczajac Shelf kolorem
pd.crosstab(cereals.Calories_binned,cereals.Shelf).plot.bar(stacked=True)


# %%
# zad 6.3 - Stworzenie znormalizowanego wykresu słupkowego dla nowej zmiennej zaznaczajac Shelf kolorem
pd.crosstab(cereals.Calories_binned,cereals.Shelf, normalize="index").plot.bar(stacked=True)


# %%



