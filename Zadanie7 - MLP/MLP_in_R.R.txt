churn<-churn_pl_dane
n<-dim(churn)

#Tworzymy nowa zmienna Indeks
churn$Index<-c(1:n[1])

#Wczytujemy ponizszy pakiet by wywolac funkcje createDataPartition
install.packages("caret")
library(caret)
#Ustawienie ziarna generatora liczb losowych w R:
set.seed(274)
#Podzial na zbior uczacy i testowy.
#Sposrod wszystkich wierszy otrzymujemy numery losowych (okolo) 70%. 
podzial<-createDataPartition(churn$Index,p=0.7, list=FALSE)

#Proba uczaca skupiajaca predyktory i zmienna Rezygnacja
train_set<-churn[podzial,c(2,3,4,5,7,8,10,11,13,14,16,17,19)]


# Zachowujemy predyktory ze zbioru uczacego (train_set) przed normalizacja 
x_train_bez_norm<-train_set[,1:12]

nor<-function(x){(x-min(x))/(max(x)-min(x))}
#Normalizujemy predyktory ze zbioru uczacego
train_set[,c(1:12)] <- lapply(train_set[,c(1:12)], nor)

#Proba testowa zawiera tylko predyktory i zmienna Rezygnacja
x_test<-churn[-podzial,c(2,3,4,5,7,8,10,11,13,14,16,17,19)]


#Normalizujemy tez zmienne ilosciowe ze zbioru testowego ale biorac minimum i maksimum z uczacego
nor2<-function(x,y){(x-min(y))/(max(y)-min(y))}
for(i in 1:12) {
x_test[,i] <-nor2(x_test[,i],x_train_bez_norm[,i])
}

# Zmienna Rezygnacja do uczenia i do testowania
y_train<-churn[podzial,19] 
y_test<-churn[-podzial,19]

install.packages("neuralnet")
library(neuralnet)
#Bardzo wazne jest by data zawieral zbior uczacy obejmujacy wszystkie predyktory 
#(w tym te znormalizowane) i zmienna celu tj Rezygnacja. Stad data=train_set.
siec_neur<-neuralnet(Rezygnacja~Czaswspolpracy+Liczbawiadomosci+Dzienminuty+Dzienrozmowy+
                       Wieczorminuty+Wieczorrozmowy+Nocminuty+Nocrozmowy+Miedzynarodoweminuty+
                       Miedzynarodowerozmowy+LiczbarozmowzBOK+Planmiedzy01,data=train_set,hidden=6,
                     act.fct = "logistic",linear.output = FALSE,stepmax=1e+05,threshold=0.1)
#Co mozna wypisac po stworzeniu sieci neuronowej
summary(siec_neur)
#np. uzyskujemy co jest zmienna celu, a co predyktorami
siec_neur$model.list 


#Narysowanie sieci
plot(siec_neur)


#By przewidziec wartosci zm. Rezygnacja w oparciu o model na probie testowej
predict=compute(siec_neur,x_test)
predict$net.result
prob<-predict$net.result
y_predict<-ifelse(prob>0.5, 1, 0)

#Macierz pomylek dla proby testowej
macierz_pom_test<-table(y_test,y_predict)
trafnosc_test<-(macierz_pom_test[1,1]+macierz_pom_test[2,2])/nrow(x_test)
czulosc_test<-(macierz_pom_test[2,2])/(macierz_pom_test[2,2]+macierz_pom_test[2,1])
swoistosc_test<-(macierz_pom_test[1,1])/(macierz_pom_test[1,1]+macierz_pom_test[1,2])
print(trafnosc_test)
print(czulosc_test)
print(swoistosc_test)


#Stosujemy uzyskany model dla danych dotyczacych nowych klientow
churn_nowe=churn_pl_dane_nowe

#Normalizujemy predyktory ilosciowe
churn_nowe[,c(2,7,8,9,11,12,14,15,17,18,20,22)] <- lapply(churn_nowe[,c(2,7,8,9,11,12,14,15,17,18,20,22)], nor)
#Predyktory ilosciowe znomralizowane+Planmiedz01
predict_nowy=compute(siec_neur,churn_nowe[,c(2,7,8,9,11,12,14,15,17,18,20,22)])
predict_nowy$net.result
prob_nowy<-predict_nowy$net.result
y_predict_nowy<-ifelse(prob_nowy>0.5, 1, 0)


#Model szacujacy liczbe minut rozmow prowadzonych przez klienta w taryfie nocnej

#Mamy predyktory: Czaswspolpracy,Liczbawiadomosci,Dzienminuty,Dzienrozmowy,
#Wieczorminuty,Wieczorrozmowy,Miedzynarodoweminuty,
# Miedzynarodowerozmowy,LiczbarozmowzBOK,Planmiedzy01 i zmienna celu Nocminuty
train_set_regr<-train_set[,c(1,2,3,4,5,6,7,9,10,11,12)]

#Tylko predyktory
x_test_regr<-x_test[,c(1,2,3,4,5,6,9,10,11,12)]

#7 zmienna to Nocminuty
y_train_regr<-train_set_regr[,7] 
y_test_regr<-x_test[,7]

#Budujemy siec
#Ponownie bardzo wazne jest by data zawieral zbior uczacy obejmujacy wszystkie predyktory 
#(w tym te znormalizowane) i zmienna celu tj Nocminuty. Stad data=train_set_regr
siec_regr<-neuralnet(Nocminuty~Czaswspolpracy+Liczbawiadomosci+Dzienminuty+Dzienrozmowy+
                     Wieczorminuty+Wieczorrozmowy+Miedzynarodoweminuty+
                     Miedzynarodowerozmowy+LiczbarozmowzBOK+Planmiedzy01,data=train_set_regr,hidden=6,
                     act.fct = "logistic",linear.output = FALSE,stepmax=1e+05,threshold=0.1)
#Rysujemy siec
plot(siec_regr)

#Otrzymane wartosci przewidywane dla Nocminuty znormalizowanej
noc_minuty_pred<-predict(siec_regr,x_test_regr,type="raw")

#Denormalizacja
k<-nrow(noc_minuty_pred)
noc_denor<-0
for(i in 1:k) {
  noc_denor[i] <- (noc_minuty_pred[i]*(max(x_train_bez_norm$Nocminuty)-min(x_train_bez_norm$Nocminuty)))+min(x_train_bez_norm$Nocminuty)
}
print(noc_denor)

#RMSE: obie zmienne nieznormalizowane. Chodzi nam o Nocminuty.
#W sczegolnosci y_test to churn[-podzial,10], bo 10 zmienna to Nocminuty
names(churn)
#RMSE
install.packages("MLmetrics")
library(MLmetrics)
print(RMSE(noc_denor, churn[-podzial,10]))

