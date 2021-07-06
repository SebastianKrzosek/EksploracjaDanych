# This Python file uses the following encoding: utf-8
import sys
import os


#from PySide2.QtWidgets import QApplication, QWidget
#from PySide2.QtCore import QFile
#from PySide2.QtUiTools import QUiLoader
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import uic

class main(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("form.ui", self)
        self.setWindowTitle("Miary Jakości")
        def initialize():
            self.label_2.setText("")
            self.tableWidget.setVisible(False)
            self.generalText.setVisible(False)
            self.generalValue.setVisible(False)
            self.columnText.setVisible(False)
            self.columnValue.setVisible(False)
            self.accuracyText.setVisible(False)
            self.accuracyValue.setVisible(False)
            self.specifyText.setVisible(False)
            self.specifyValue.setVisible(False)
            self.tendernessText.setVisible(False)
            self.tendernessValue.setVisible(False)


        def is_number(string):
            try:
                float(string)
                return True
            except ValueError:
                return False

        def allNumeric():
            for i in range(int(self.textK.toPlainText())):
                for j in range(int(self.textK.toPlainText())):
                    if not self.tableWidget.item(i,j):
                        return False
                    elif not self.tableWidget.item(i,j).text():
                        return False
                    elif not is_number(self.tableWidget.item(i,j).text()):
                        return False
            return True

        def setTable():
            initialize()
            if self.textK == "":
                self.tableWidget.setVisible(False)
            if not self.textK.toPlainText().isnumeric():
                self.textK.setText("")
                self.tableWidget.setVisible(False)
            else:
                self.tableWidget.resizeColumnsToContents()
            if self.textK.toPlainText().isnumeric():
                if int(self.textK.toPlainText()) > 1:
                    self.tableWidget.setVisible(True)
                    if int(self.textK.toPlainText()) > 10:
                        self.textK.setText("10")

                    self.tableWidget.setRowCount(int(self.textK.toPlainText()))
                    self.tableWidget.setColumnCount(int(self.textK.toPlainText()))

                if int(self.textK.toPlainText()) == 2:
                    self.tableWidget.setMinimumWidth(230)
                    self.tableWidget.setMaximumWidth(230)
                elif int(self.textK.toPlainText()) * 100 < 1150:
                    self.tableWidget.setMinimumWidth(int(self.textK.toPlainText()) * 110)
                    self.tableWidget.setMaximumWidth(int(self.textK.toPlainText()) * 110)
                else:
                    self.tableWidget.setMinimumWidth(1150)
                    self.tableWidget.setMaximumWidth(1150)
                if int(self.textK.toPlainText()) * 60 < 400:
                    self.tableWidget.setMinimumHeight(int(self.textK.toPlainText()) * 60)
                    self.tableWidget.setMaximumHeight(int(self.textK.toPlainText()) * 60)
                else:
                    self.tableWidget.setMinimumHeight(400)
                    self.tableWidget.setMaximumHeight(400)
            if self.textK.toPlainText().isnumeric():
                if int(self.textK.toPlainText()) < 2:
                    self.label_2.setText("Wprowadź wartość > 1 i 10 >")

        def clickColumn(index):
            if allNumeric():
                temp_matrix = [[0, 0],[0, 0]]
                sum = 0
                other_first = 0
                other_second = 0
                for i in range(int(self.textK.toPlainText())):
                    for j in range(int(self.textK.toPlainText())):
                         sum += int(self.tableWidget.item(i,j).text())
                         if i == j and i == index:
                             temp_matrix[1][1] = int(self.tableWidget.item(i,j).text())
                         elif i == index and i != j:
                             temp_matrix[1][0] += int(self.tableWidget.item(i,j).text())
                         elif j == index and i != j:
                             temp_matrix[0][1] += int(self.tableWidget.item(i,j).text())

                temp_matrix[0][0] = sum - (temp_matrix[1][1] + temp_matrix[1][0] + temp_matrix[0][1])

                myAccuracy = round((100 * (temp_matrix[0][0] + temp_matrix[1][1])/sum),2)
                myTenderness = round((100 * temp_matrix[0][0] / (temp_matrix[0][0] + temp_matrix[0][1])), 2)
                mySpecify =  round((100 * temp_matrix[1][1]/(temp_matrix[1][0] + temp_matrix[1][1])),2)


                self.accuracyValue.setText(str(myAccuracy) + "%")
                self.specifyValue.setText(str(myTenderness) + "%")
                self.tendernessValue.setText(str(mySpecify) + "%")

                self.columnText.setVisible(True)
                self.columnValue.setVisible(True)
                self.accuracyText.setVisible(True)
                self.accuracyValue.setVisible(True)
                self.specifyText.setVisible(True)
                self.specifyValue.setVisible(True)
                self.tendernessText.setVisible(True)
                self.tendernessValue.setVisible(True)
                self.columnValue.setText(str(index+1))
            else:
                initialize()

        def tableChanged():
            if allNumeric():
                self.label_2.setText("")
                self.generalText.setVisible(True)
                self.generalValue.setVisible(True)
                sum = 0
                result = 0
                for i in range(int(self.textK.toPlainText())):
                    for j in range(int(self.textK.toPlainText())):
                        sum += int(self.tableWidget.item(i,j).text())
                        if i==j:
                            result += int(self.tableWidget.item(i,j).text())
                self.generalValue.setText(str(round((result/sum) * 100,2) ) + "%")

            else:
                initialize()
                self.label_2.setText("Uzupełnij komórki.")

        def connectEvents():
            self.textK.textChanged.connect(setTable)
            self.pushButton.clicked.connect(tableChanged)
            self.tableWidget.horizontalHeader().sectionClicked.connect(clickColumn)

        initialize()
        connectEvents()


if __name__ == "__main__":
    app = QApplication([])
    widget = main()
    widget.show()
    sys.exit(app.exec_())
