#Najpierw importujemy baze a potem zapisujemy jako adult
adult=adult_ch3_training
n<-dim(adult)

#Zmienne z kategoriami musza miec typ factor
adult$marital.status<-factor(adult$marital.status)
adult$workclass<-factor(adult$workclass)
adult$occupation<-factor(adult$occupation)
adult$sex<-factor(adult$sex)
adult$income<-factor(adult$income)

#Wczytujemy poni¿szy pakiet by wywolac funkcje createDataPartition
install.packages("caret")
library(caret)

#Tworzymy nowa zmienna Indeks
adult$Index<-c(1:n[1])

#Ustawienie ziarna generatora liczb losowych w R:
set.seed(123)
#Podzial na zbior uczacy i testowy.
#Sposrod wszystkich wierszy otrzymujemy numery losowych (okolo) 70%. 
podzial<-createDataPartition(adult$Index,p=0.7, list=FALSE)

x_train<-adult[podzial,c(1:8)]
x_test<-adult[-podzial,c(1:8)]

y_train<-adult[podzial,9] 
y_test<-adult[-podzial,9]

t<-table(adult$income)
proporcja=c(round(t[1]/n[1],2),round(t[2]/n[1],2))
print(proporcja)
#Mo¿emy zobaczyæ ile jest w zbiorach uczacym i testowym
nrow(x_train)
nrow(x_test)

#Instalujemy i ladujemy pakiet do tworzenia drzew C50
install.packages("C50") #Uwaga du¿a litera C
library(C50)

C50<-C5.0(income~age+workclass+education+marital.status+occupation+sex+capital.gain+capital.loss,
          data=adult[podzial,],control=C5.0Control(minCases = 75))
#Minimlna liczba rekordow w lisciu = 75 
#Wywolujac summary dostajemy OBB-calkowity wspolczynnik bledu,
#Dostajemy macierz pomylek
summary(C50)
#Rysujemy drzewo
plot(C50)


y_predict_C50<-predict(C50,x_test)
#lub opcojnalnie z prawdopodobienstwami
#y_predict_C50<-predict(C50, x_test,type="prob")

zestawienie_c50=data.frame(y_test,y_predict_C50)


#Macierz pomylek dla proby testowej
macierz_pom1<-table(y_test,y_predict_C50)
trafnosc_C50<-(macierz_pom1[1,1]+macierz_pom1[2,2])/nrow(x_test)
czulosc_C50<-(macierz_pom1[2,2])/(macierz_pom1[2,2]+macierz_pom1[2,1])
swoistosc_C50<-(macierz_pom1[1,1])/(macierz_pom1[1,1]+macierz_pom1[1,2])
print(trafnosc_C50)
print(czulosc_C50)
print(swoistosc_C50)

#Krzywa ROC dla proby testowej
install.packages("ROCR")
library(ROCR)
predict_wykres_C50<-prediction(c(y_predict_C50),y_test)
#"sens"(sensitivity)=czulosc, "fpr"(false positive rate)=1-swoistosc 
wykres_ROC_1<-performance(predict_wykres_C50,"sens","fpr")
windows(5,5)
par(mar=c(4,4,1.5,0.2))
plot(wykres_ROC_1,xlab="1-Swoistosc", ylab="Czulosc")
abline(0,1,lty=2)#linia referencyjna przerywana (lty=2)


#Instalujemy i ladujemy pakiet do tworzenia lasów loswych
install.packages("randomForest")
library(randomForest)

las<-randomForest(income~age+workclass+education+marital.status+occupation+sex+capital.gain+capital.loss,
                               data=adult[podzial,],ntree=100)
#Wywoujac las dostajemy OBB-calkowity wspolczynnik bledu,
#Dostajemy macierz pomylek
#Dostajemy wskaznik falszywie pozytywnych i wskaznik falszywie negatywnych
las
#Wykres istotnosci zmiennych wykonany za pomoca funkcji varImpPlot.
importance(las)
windows(5,5)
par(mar=c(4,4,1.5,0.2))
varImpPlot(las)

#Wykres b³edu klasyfikacji w zaleznosci od liczby drzew.
windows(5,5)
par(mar=c(4,4,1.5,0.2))
plot(las)
leg <- c("<=50k",">50k","OBB" )
legend("center",legend=leg, col=c("red","green","black"), lty=1,
         pch=20, cex=0.7,inset=0)

y_predict_las<-predict(las, x_test)
#lub opcjonalnie z prawdopodobienstwami
#y_predict_las<-predict(las, x_test,type="prob")

zestawienie_las=data.frame(y_test,y_predict_las)

#Macierz pomylek dla proby testowej
macierz_pom2<-table(y_test,y_predict_las)
trafnosc_las<-(macierz_pom2[1,1]+macierz_pom2[2,2])/nrow(x_test)
czulosc_las<-(macierz_pom2[2,2])/(macierz_pom2[2,2]+macierz_pom2[2,1])
swoistosc_las<-(macierz_pom2[1,1])/(macierz_pom2[1,1]+macierz_pom2[1,2])
print(trafnosc_las)
print(czulosc_las)
print(swoistosc_las)

#Krzywa ROC dla proby testowej (pakiet zaladowany wczesniej)
predict_wykres_las<-prediction(c(y_predict_las),y_test)
#"sens"(sensitivity)=czulosc, "fpr"(false positive rate)=1-swoistosc 
wykres_ROC_2<-performance(predict_wykres_las,"sens","fpr")
windows(5,5)
par(mar=c(4,4,1.5,0.2))
plot(wykres_ROC_2,xlab="1-Swoistosc", ylab="Czulosc")
abline(0,1,lty=2)#linia referencyjna przerywana (lty=2)