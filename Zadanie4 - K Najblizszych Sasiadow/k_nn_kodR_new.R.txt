#£adujemy baze iris
data(iris)
#Potrzebne nazwy zmiennych
names(iris)
#Wymiar bazy danych
n<-dim(iris)
#Standaryzacja predyktorow min- max (definicja funkcji)
stand<-function(x){(x-min(x))/(max(x)-min(x))}
#Standaryzujemy pierwsze kolumny z bazy danych iris.
#Dlugosc, szerokosc platkow korony i dzialek kielicha to predyktory.
#Funkcja lapply() pozwala na wykonanie okreslonego dzialania (u nas standaryzacji),
#na kazdym z wymienionych elementow.
#Funkcja as.data.frame konwersuje na ramke danych.
iris_stand <- as.data.frame(lapply(iris[,c(1,2,3,4)], stand))

#Podzial na zbior uczacy i testowy.
#Tworzymy nowa zmienna Indeks
iris$Index<-c(1:n[1])

#Ustawienie ziarna generatora liczb losowych w R:
set.seed(5)

#Wczytujemy poni¿szy pakiet by wywolac funkcje createDataPartition
install.packages("caret")
library(caret)

podzial<-createDataPartition(iris$Index,p=0.7, list=FALSE)

x_train<-iris[podzial,c(1:4)]
x_test<-iris[-podzial,c(1:4)]

y_train<-iris[podzial,5] 
y_test<-iris[-podzial,5]

#Mo¿emy zobaczyæ ile jest w zbiorach uczacym i testowym
nrow(x_train)
nrow(x_test)

#Instalujemy pakiet i otwieramy jego biblioteke
install.packages("class")
library(class)

#Uruchamiamy algorytm
y_pred<-knn(x_train,x_test,y_train,k=3,prob=TRUE)
#Zestawienie wartosci rzeczywistych z przewidywanymi i prawdopodobienstwami
zestawienie<-data.frame(y_test,y_pred,attr(y_pred,"prob"))

#Macierz pomylek
tab<-table(y_pred,y_test)


#Pomocniczo, by uzyc funkcji revalue
install.packages("plyr")
library(plyr)
#Krzywe ROC 
install.packages("ROCR")
library(ROCR)

#krzywa ROC dla setosa
pred_setosa<-c(revalue(x=y_pred,replace=c("setosa"=1,"versicolor"=0,"virginica"=0)))
y_test_setosa<-c(revalue(x=y_test,replace=c("setosa"=1,"versicolor"=0,"virginica"=0)))
predict_setosa<-prediction(pred_setosa,y_test_setosa)
#"sens"(sensitivity)=czulosc, "fpr"(false positive rate)=1-swoistosc 
perf_setosa<-performance(predict_setosa,"sens","fpr")
windows(5,5)
par(mar=c(4,4,1.5,0.2))
plot(perf_setosa)

#krzywa ROC dla versicolor
pred_versicolor<-c(revalue(x=y_pred,replace=c("setosa"=0,"versicolor"=1,"virginica"=0)))
y_test_versicolor<-c(revalue(x=y_test,replace=c("setosa"=0,"versicolor"=1,"virginica"=0)))
predict_versicolor<-prediction(pred_versicolor,y_test_versicolor)
perf_versicolor<-performance(predict_versicolor,"sens","fpr")
windows(5,5)
par(mar=c(4,4,1.5,0.2))
plot(perf_versicolor)


#krzywa ROC dla virginica
pred_virginica<-c(revalue(x=y_pred,replace=c("setosa"=0,"versicolor"=0,"virginica"=1)))
y_test_virginica<-c(revalue(x=y_test,replace=c("setosa"=0,"versicolor"=0,"virginica"=1)))
predict_virginica<-prediction(pred_virginica,y_test_virginica)
perf_virginica<-performance(predict_virginica,"sens","fpr")
windows(5,5)
par(mar=c(4,4,1.5,0.2))
plot(perf_virginica)



