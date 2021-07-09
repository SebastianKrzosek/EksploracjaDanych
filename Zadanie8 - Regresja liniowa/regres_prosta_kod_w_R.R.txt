# Ustawienie wielkosci marginesow dla wykresu
windows(5,5)
par(mar=c(4,4,1.5,0.2))
#Wykres rozrzutu przedstawiajacy zaleznosc kalorycznosci piwa od zawartosci alkoholu
plot(piwo$alkohol,piwo$kalorie,main="Wykres zaleznosci kalorycznosci piwa od zawartosci alkoholu", 
     col ="blue",pch=21,cex=1.5,xlab="Alkohol",ylab="Kalorie")

#Wpsolczynnik korelacji kalorii i alkoholu
R=cor(piwo$kalorie,piwo$alkohol)
print(R)

#Model regresji liniowej
model1<-lm( kalorie ~ alkohol, data=piwo)

#Wspolczynniki rownania regresji
beta1<-coef(model1)
print(beta1)

#Wpsolczynnik R^2 mozna tez policzyc tak:
podsumowanie1<-summary(model1)
podsumowanie1$r.squared


#Wartosci zaobserwowane (na niebiesko) i dopasowane (na czerwono) na wspolnym wykresie.
windows(5,5)
par(mar=c(4,4,0.2,0.2))
plot(piwo$alkohol,piwo$kalorie,col ="blue",pch=21,cex=1.5,xlab="Alkohol",ylab="Kalorie",ylim=c(10,60))
points(piwo$alkohol,fitted(model1), col="red", pch=21)
#Dodana prosta regresji z wspolczynnikami beta1
abline(model1, col="red")

#Bledy oszacowan, tzw. wartosci resztowe
piwo$reszty<-resid(model1)

#Wykresy dla reszt
windows(5,5)
par(mar=c(4,4,0.2,0.2))
hist(resid(model1), main="Histogram dla reszt",col=rainbow(20))

windows(5,5)
par(mar=c(4,4,0.2,0.2))
plot(model1,which=2,labels.id = names(residuals(model1)))


install.packages("stats")
library(stats)
#Test normalnosci dla reszt, jesli p-value>0.05, to mozna przyjac, ze maja rozklad normalny
#Potrzebny pakiet stats
shapiro.test(piwo$reszty)

#Wykonujemy Durbin-Watson test - potrzebny pakiet car
#H0 (null hypothesis): There is no correlation among the residuals.
# Jest to test niezaleznosci dla reszt. Jesli p-value>0.05, to mozna przyjac, ze
#reszty sa niezalezne
library(car)
durbinWatsonTest(model1)


#Na wykresie rozrzutu przedstawia zaleznosc reszt standaryzowanych wzgledem 
#standaryzowanych wartosci przewidywanych
windows(5,5)
par(mar=c(4,4,0.2,0.2))
plot(scale(fitted(model1)),rstandard(model1),ylab="Standaryzowane reszty",xlab="Standaryzowane wartosci przewdidywane")


#Identyfikacja obserwacji wplywowych (oznaczone sa *)
influence.measures(model1)

#Identyfikacja obserwacji odstajacych
install.packages("car")
library(car)
outlierTest(model1)

#Nowa zmienna o wartosciach 1, jesli piwo jest light, i 0, jesli nie
piwo$znacznik<-0
for(i in 1:length(piwo$alkohol)) {
  if (piwo$alkohol[i]<=4.3){
    piwo$znacznik[i]=1
  }
  else{
    piwo$znacznik[i]=0
  }
}

#Model regresji liniowej dla light
model2<-lm( kalorie[which(piwo$znacznik==1)] ~ alkohol[which(piwo$znacznik==1)], data=piwo)

#Wspolczynniki rownania regresji
beta2<-coef(model2)
print(beta2)

#Wartosci zaobserwowane (na niebiesko) i dopasowane (na czerwono) na wspolnym wykresie.
windows(5,5)
par(mar=c(4,4,0.2,0.2))
plot(piwo$alkohol[which(piwo$znacznik==1)],piwo$kalorie[which(piwo$znacznik==1)],col ="blue",pch=21,cex=1.5,xlab="Alkohol",ylab="Kalorie",ylim=c(10,60))
points(piwo$alkohol[which(piwo$znacznik==1)],fitted(model2), col="red", pch=21)
#Dodana prosta regresji z wspolczynnikami beta2
abline(model2, col="red")

#Wpsolczynnik R^2:
podsumowanie2<-summary(model2)
podsumowanie2$r.squared

#Na wykresie rozrzutu przedstawia zaleznosc reszt standaryzowanych wzgledem 
#standaryzowanych wartosci przewidywanych
windows(5,5)
par(mar=c(4,4,0.2,0.2))
plot(scale(fitted(model2)),rstandard(model2),ylab="Standaryzowane reszty",xlab="Standaryzowane wartosci przewdidywane")


#Model regresji liniowej dla nie light
model3<-lm(kalorie[which(piwo$znacznik==0)] ~ alkohol[which(piwo$znacznik==0)], data=piwo)

#Wspolczynniki rownania regresji
beta3<-coef(model3)
print(beta3)

#Wartosci zaobserwowane (na niebiesko) i dopasowane (na czerwono) na wspolnym wykresie.
windows(5,5)
par(mar=c(4,4,0.2,0.2))
plot(piwo$alkohol[which(piwo$znacznik==0)],piwo$kalorie[which(piwo$znacznik==0)],col ="blue",pch=21,cex=1.5,xlab="Alkohol",ylab="Kalorie",ylim=c(10,60))
points(piwo$alkohol[which(piwo$znacznik==0)],fitted(model3), col="red", pch=21)
#Dodana prosta regresji z wspolczynnikami beta2
abline(model3, col="red")

#Wpsolczynnik R^2:
podsumowanie3<-summary(model3)
podsumowanie3$r.squared

#Na wykresie rozrzutu przedstawia zaleznosc reszt standaryzowanych wzgledem 
#standaryzowanych wartosci przewidywanych
windows(5,5)
par(mar=c(4,4,0.2,0.2))
plot(scale(fitted(model3)),rstandard(model3),ylab="Standaryzowane reszty",xlab="Standaryzowane wartosci przewdidywane")

#Wywolanie modelu na nowym zbiorze danych
#nowe_piwa<-data.frame(alkohol=c(wartoci alkoholu))
#predict(nazwa stworzonego modelu, newdata = nowe dane na ktorych model ma dzialac)
nowe_piwa<-data.frame(alkohol=c(6.5,3,2.5,4.5))
predict(model1,newdata=nowe_piwa)