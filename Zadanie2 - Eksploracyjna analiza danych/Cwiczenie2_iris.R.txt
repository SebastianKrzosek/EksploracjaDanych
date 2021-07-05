#Macierz korelacji
my_data<-iris[, c(1:5)]
macierz_korel<- round(cor(my_data),2)
print(round(macierz_korel, 2))

#Macierz wykresow rozrzutow
install.packages("car")
library(car)
windows(5,5)
par(mar=c(4,4,1.5,0.2))
pairs(~klasa+dlugosc_kielich+szerokosc_kielich+dlugosc_platek+szerokosc_platek,data=iris,
      main="Macierz wykresow rozrzutow")

#Trojwymiarowy wykres rozrzutu 
install.packages("scatterplot3d")
library(scatterplot3d)
windows(5,5)
par(mar=c(4,4,1.5,0.2))
s3d<-scatterplot3d(iris$dlugosc_kielich,iris$dlugosc_platek,iris$klasa,xlab='Dlugosc kielicha',ylab='Dlugosc platka',zlab='Gatunek', main="3D wykres rozrzutu")

