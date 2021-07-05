bank_train<-bank_marketing_training

# Sprawdzamy ile rekordów i zmiennych ma zaimportowany plik. 
# Pierwsza wspolrzedna okresla liczbe obserwacji/rekordow, a druga liczbe zmiennych.
n<-dim(bank_train)

#Tworzymy nowa zmienna Indeks
bank_train$Index<-c(1:n[1])

#Obejrzymy nazwy zmiennych, wraz z 6 poczatkowymi obserwacjami.
head(bank_train)

#Rysujemy histogram zmiennej days_since_previous.
windows(5,5)
par(mar=c(4,4,1.5,0.2))
hist(bank_train$days_since_previous,xlab="days_since_previous",
     main="Histogram of days_since_previous")

# Zamieniamy 999 na kod braku danych Na.
bank_train$days_since_previous<-ifelse(test=bank_train$days_since_previous==999,yes=NA,no=bank_train$days_since_previous)

#Rysujemy histogram zmiennej days_since_previous po zmianie.
windows(5,5)
par(mar=c(4,4,1.5,0.2))
hist(bank_train$days_since_previous,xlab="days_since_previous",
     main="Histogram of days_since_previous-Missing Values replaced by NA")


# Rysujemy wykres supkowy dla zmiennej bank_train$education
barplot(table(bank_train$education),horiz=TRUE)
# Importujemy pakiet plyr.
install.packages("plyr")
# i ladujemy jego biblioteke.
library(plyr)

#Definiujemy s³ownik, który zast¹pi kategorie zmiennej education_numeric liczbami.
edu.num<-revalue(x=bank_train$education,replace=c("illiterate"=0,"basic.4y"=4,"basic.6y"=6,"basic.9y"=9,"high.school"=12,
                                                   "professional.course"=12,"university.degree"=16,"unknown"=NA))

#Zobaczmy ten sownik. rekordy traktowane jako zmienne lancuchowe
edu.num

#Nie mozna narysowac histogramu.
hist(edu.num)

#Tworzymy nowa zmienna education_numeric, gdzie kategorie sa juz liczbami a nie lancuchami liczb.
bank_train$education_numeric<-as.numeric(levels(factor(edu.num)))[factor(edu.num)]

#Ju¿ teraz histogram powstanie
windows(5,5)
par(mar=c(4,4,1.5,0.2))
hist(bank_train$education_numeric)


# Standaryzujemy zmienna age.
bank_train$age_z<-scale(x=bank_train$age)

# Identyfikujemy obserwacje odstajace o co najmniej 3 odchylenia standardowe.
bank_outliers<-bank_train[which(bank_train$age_z < -3|bank_train$age_z > 3),]

#Sortujemy baze bank_train malejaca (stad minus w srodku) ze wzgledu na wiek.
bank_train_sort<-bank_train[order(-bank_train$age_z),]

#Wypisujemuy 10 rekordów najstraszych.
bank_train_sort[1:10,]

#Wyswietla 6 pierwszych rekordow (tak domyslnie ma ustawione).
head(bank_train_sort)

#Wyswietla dla 10 pierwszych rekordow ich wiek i stan cywilny.
bank_train_sort[1:10,c(1,3)]



