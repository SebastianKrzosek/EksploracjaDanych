#wczytywanie danych
data = bank_marketing_training
#ustawienie indeksu jako ziarno. 
set.seed(300136)

#zamiana zmiennej celu response z yes i no -> 1 i 0
data[names(data)[3]]  = ifelse(data[names(data)[3]] == "married", 1, 0)
data[names(data)[5]]  = ifelse(data[names(data)[5]] == "yes", 1, ifelse(data[names(data)[5]] == "no",0 , 999))
data[names(data)[6]]  = ifelse(data[names(data)[6]] == "yes", 1, ifelse(data[names(data)[6]] == "no",0 ,999))
data[names(data)[7]]  = ifelse(data[names(data)[7]] == "yes", 1, ifelse(data[names(data)[7]] == "no",0 ,999))
data[names(data)[15]]  = ifelse(data[names(data)[15]] == "success", 1, ifelse(data[names(data)[15]] == "failure", 0 ,999))
data[names(data)[21]] = ifelse(data[names(data)[21]] == "no", 0, 1)
data <- data[, sapply(data, is.numeric)]

#dodanie indeksu
n<-dim(data)
data$Index<-c(1:n[1])

#przygotowanie podzialu
#install.packages("caret")
library(caret)
podzial<-createDataPartition(data$Index, p=0.7, list=FALSE)

#Macierz korelacji 
#install.packages("corrplot")
library(corrplot)
corrplot(cor(data), method="number")



#===================================#
          #algorytm K-NN
#===================================#

#install.packages("class")
library(class)
#standaryzacja min-max
stand<-function(x){(x-min(x))/(max(x)-min(x))}
data_stand <- as.data.frame(lapply(data, stand))
data_stand$Index = data["Index"]

#podzielenie zbioru na testowy i uczacy
#zmienne silnie skorelowane zostaly odrzucone
x_train<-data_stand[podzial,c(1,2,3,5,6,7,8,9,10,11,12,13)]
x_test<-data_stand[-podzial,c(1,2,3,5,6,7,8,9,10,11,12,13)]
y_train<-data_stand[podzial,16] 
y_test<-data_stand[-podzial,16]

#uruchomienie algorytmu
knn_test<-knn(x_train,x_test,y_train,k=29,prob=TRUE)
knn_train<-knn(x_train,x_train,y_train,k=29,prob=TRUE)

#Zestawienie wyniku z wartoscia rzeczywistymi i prawdopodobienstwami
zestawienie<-data.frame(y_test,knn_test,attr(knn_test,"prob"))
zestawienie
#Macierz pomylek dla proby uczacej
tab_train<-table(knn_train,y_train)
czulosc_train<-(tab_train[2,2])/(tab_train[2,2]+tab_train[2,1])
trafnosc_train<-(tab_train[1,1]+tab_train[2,2])/nrow(x_train)
swoistosc_train<-(tab_train[1,1])/(tab_train[1,1]+tab_train[1,2])
#Macierz pomylek dla proby testowej
tab_test<-table(knn_test,y_test)
czulosc_test<-(tab_test[2,2])/(tab_test[2,2]+tab_test[2,1])
trafnosc_test<-(tab_test[1,1]+tab_test[2,2])/nrow(x_test)
swoistosc_test<-(tab_test[1,1])/(tab_test[1,1]+tab_test[1,2])
#Miary jakosci
czulosc_test
czulosc_train
trafnosc_test
trafnosc_train
swoistosc_test
swoistosc_train

library(ROCR)
predict_knn<-prediction(y_test,c(knn_test))
ROC_1<-performance(predict_knn,"sens","fpr")
plot(ROC_1,xlab="1-Swoistosc", ylab="Czulosc")
abline(0,1,lty=2)
                      


#===================================#
                #C5.0
#===================================#


#obliczenie trafnosci sposobem naiwanym, 
#korzystanie z algorytmu powinno zwrocic lepszy wynik
tab<-table(data$response)
proporcja=c(round(tab[1]/n[1],2),round(tab[2]/n[1],2))
proporcja

#podobnie jak w knn
#zmienne silnie skorelowane zostaly odrzucone
x_train<-data[podzial,c(1,2,3,5,6,7,8,9,10,11,12,13)]
x_test<-data[-podzial,c(1,2,3,5,6,7,8,9,10,11,12,13)]
#przypisanie zmiennej celu - (response)
y_train<-data[podzial,16] 
y_test<-data[-podzial,16]



#install.packages("C50")
library(C50)
#budowa modelu klasyfikujacego C5.0
C50<-C5.0(as.factor(response)~age+marital+default+loan+duration+
            campaign+days_since_previous+previous+
            previous_outcome+emp.var.rate+cons.price.idx+cons.conf.idx,
            data=data[podzial,], control= C5.0Control(minCases = 50))

#C50<-C5.0(as.factor(response)~age+marital+duration+campaign+days_since_previous+previous+
#            emp.var.rate+cons.price.idx+cons.conf.idx,
#            data=data[podzial,], control= C5.0Control(minCases = 50))


#Dostajemy macierz pomylek, dzieki czemu mozemy sprawdzic, 
#ktore predyktory model uznal‚ za najwazniejsze
summary(C50)

#Rysujemy drzewo
plot(C50)

#Sprawdzenie na danych testowych
y_pred_c50_test<-predict(C50,x_test)
#Sprawdzenie na danych treningowych
y_pred_c50_train<-predict(C50,x_train)

#Zestawienie y_test/y_train z przewidywanymi
zest_c50_test=data.frame(y_test,y_pred_c50_test)
zest_c50_train=data.frame(y_train,y_pred_c50_train)

#Macierz pomylek dla proby testowej
conf_mat_test<-table(y_test,y_pred_c50_test)
czulosc_test<-(conf_mat_test[2,2])/(conf_mat_test[2,2]+conf_mat_test[2,1])
trafnosc_test<-(conf_mat_test[1,1]+conf_mat_test[2,2])/nrow(x_test)
swoistosc_test<-(conf_mat_test[1,1])/(conf_mat_test[1,1]+conf_mat_test[1,2])
#Macierz pomylek dla proby uczacej
conf_mat_train<-table(y_train,y_pred_c50_train)
czulosc_train<-(conf_mat_train[2,2])/(conf_mat_train[2,2]+conf_mat_train[2,1])
trafnosc_train<-(conf_mat_train[1,1]+conf_mat_train[2,2])/nrow(x_train)
swoistosc_train<-(conf_mat_train[1,1])/(conf_mat_train[1,1]+conf_mat_train[1,2])

czulosc_test
czulosc_train
trafnosc_test
trafnosc_train
swoistosc_test
swoistosc_train

#Krzywa ROC dla proby testowej
#install.packages("ROCR")
#library(ROCR)
predict_wykres_C50<-prediction(c(y_pred_c50_test),y_test)
ROC_1<-performance(predict_wykres_C50,"sens","fpr")
plot(ROC_1,xlab="1-Swoistosc", ylab="Czulosc")
abline(0,1,lty=2)#linia referencyjna przerywana (lty=2)

