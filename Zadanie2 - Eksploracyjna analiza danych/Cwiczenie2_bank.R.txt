#Otwieramy bazê danych bank_marketing_training i zapisujemy jako bank_train.
bank_train<-bank_marketing_training

#Instalujemy pakiet ggplot2
install.packages("ggplot2")
#Ladujemy jego bibliotekê.
library(ggplot2)

#Rysujemy wykres s³upkowy (s³upki poziomo dziêki coord_flip()) dla zmiennej previous_outcome. 
ggplot(bank_train,aes(previous_outcome))+geom_bar()+coord_flip()
#Wykonujemy zestawiony wykres s³upkowy. Zestawiamy zmienn¹ previous_outcome z response.
ggplot(bank_train,aes(previous_outcome))+geom_bar(aes(fill=response))+coord_flip()
#Rysujemy znormalizowanaw¹ wersjê poprzedniego histogramu. Wszystkie s³upki maj¹ t¹ sam¹ wysokosc i szerokosc.
ggplot(bank_train,aes(previous_outcome))+geom_bar(aes(fill=response),position="fill")+coord_flip()



#Wykonujemy tabelê krzy¿ow¹ zestawiaj¹c wartosci zmiennej response (w wierszach) i previous_outcome (w kolumnach).
t.v1<-table(bank_train$response,bank_train$previous_outcome)
#Do stworzonej przed chwil¹ tabeli (A=t.v1) dodajemy wiersz i kolumnê o nazwie Total, w której
#pojawi¹ siê sumy elementów z odpowiednich wierszy i kolumn.
t.v2<-addmargins(A=t.v1,FUN=list(total=sum),quiet=TRUE)
#Tworzymy now¹ tabelê z odsetkami zamiast liczebnosci (polecenie port.table), odsetki liczone s¹ w kolumnach (margin=2).
#Gdyby mia³y byæ w weirszach, ustawiamy margin=1.
#Wyniki w tabeli zapisujemy jako procenty zaokr¹glone do 1 miejsca po przecinku (st¹d w poleceniu 1).
round(prop.table(t.v1,margin=2)*100,1)



#Rysujemy histogram dla zmiennej age. Parametr color="black" powoduje, ¿e obwoluty s³upków s¹ czarne.
ggplot(bank_train,aes(age))+geom_histogram(color="black")
#Rysujemy histogram, gdzie zmienna age bêdzie zestawiona ze zmienn¹ response.
ggplot(bank_train,aes(age))+geom_histogram(aes(fill=response),color="black")
#Rysujemy znormalizowanaw¹ wersjê poprzedniego histogramu. Wszystkie s³upki maj¹ t¹ sam¹ wysokosc i szerokosc.
ggplot(bank_train,aes(age))+geom_histogram(aes(fill=response),color="black",position="fill")



#Dzielimy zmienn¹ age na kategorie wiekowe. Musimy daæ 60,01, ¿eby przedzia³ zawiera³ 60.Mamy przedzialy
#[0,27),[27,60.01),[60.01,100)
bank_train$age_binned<-cut(x=bank_train$age,breaks=c(0,27,60.01,100),right=FALSE,
                           labels=c("Under 27","27 to 60","Over 60"))
#Bêdziemy rysowaæ wykres s³upkowy zestawiony dla zmiennej age_binned wraz z response.
ggplot(bank_train,aes(age_binned))+geom_bar(aes(fill=response))+coord_flip()
#Rysujemy znormalizowanaw¹ wersjê poprzedniego wykresu. Wszystkie s³upki maj¹ t¹ sam¹ wysokosc i szerokosc.
ggplot(bank_train,aes(age_binned))+geom_bar(aes(fill=response),position="fill")+coord_flip()
#Wykonujemy tabelê krzy¿ow¹ zestawiaj¹c wartosci zmiennej response i age_binned. 
t2<-table(bank_train$response,bank_train$age_binned)
#Wartosci w tabeli zapisujemy w procentach z dokadnosci¹ do 1 miejsca po przecinku.
t2<-round(prop.table(t2,margin=2)*100,1)