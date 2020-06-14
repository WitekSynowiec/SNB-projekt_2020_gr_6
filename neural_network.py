
import numpy as np
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import random

# Funkcja pochodna sigmoidy
def sigmoid_der(xx):
    return expit(-xx) * (1 - expit(-xx))

class NeuralNetwork:
    def __init__(self, numberOfAttempts = 15, maxNumberOfEpochs = 50, eta = 0.05, numberOfLayers = 3):
        """
        numberOfLayers - wliczamy w to wszystkie warstwy (warstwe wejsciowa, warstwy ukryte, warstwe wyjsciowa)
        """
        self.numberOfAttempts = numberOfAttempts
        self.maxNumberOfEpochs = maxNumberOfEpochs

        self.trainingData = np.genfromtxt('trainingData.csv', delimiter=',')
        #self.trainingData = np.delete(self.trainingData,0,0)

        self.eta = eta
        self.numberOfHiddenLayers = numberOfLayers - 2

        self.weightsOfNeuralNetwork = []

    def initRandomWeights(self):
        """
        Inicjalizuje losowe wagi polaczen w sieci. 
        """
        # Ustawienie małych, losowych wag z przedziału [-1,1]
        # Inicjacja wag warstw ukrytych
        # self.weights[0] - to wagi dla 1. warstwy ukrytej, itd. a ostatni element to wagi warstwy wyjsciowej 
        self.weights = {}
        for i in range(self.numberOfHiddenLayers):
            self.weights[i] = 2 * np.random.rand(np.size(self.trainingData, 1) - 1, np.size(self.trainingData, 1) - 1) - 1

        # Inicjacja wag warstwy wyjsciowej
        self.weights[len(self.weights)] = 2 * np.random.random(np.size(self.trainingData, 1) - 1) - 1

    def teachNeuralNetwork(self):
        """
        Uczy siec neuronowa dopoki procent poprawnych odpowiedzi nie bedzie wynosil conajmniej 98% lub zostanie przekroczona maksymalna liczba epochs.
        """
        for numberOfAttempt in range(self.numberOfAttempts):
            print("PROBA {0}.".format(numberOfAttempt+1))
            self.initRandomWeights()

            # Obliczanie bledu MSE oraz % poprawnie zakwalifikowanych dla poczatkowej wartosci wag dla calego zbioru treningowego.
            correctlyQualified, MSE = self.countErrorsInZeroEpoch()
            vecOfMSE = [MSE]
            vecOfCorrectlyQualified = [correctlyQualified]

            correctlyQualified = 0.0
            numberOfCurrentEpoch = 0

            while (correctlyQualified <= 0.98) and (numberOfCurrentEpoch < self.maxNumberOfEpochs):
                numberOfCurrentEpoch += 1

                # Tworzymy losowa kolejnosc wierszy ktore bedziemy brac do uczenia
                orderOfRows = list(range(np.size(self.trainingData,0)))
                random.shuffle(orderOfRows)
            
                numberOfCorrectlyQualified = 0
                MSE = 0.0

                # Przechodzimy kazdy wiersz ze zbioru treningowego w losowej kolejnosci
                for i in range(len(orderOfRows)):
                    currentRow = self.trainingData[orderOfRows[i]]              

                    # Wartosci pobudzen oraz wyjsc wszystkich neuronow w sieci
                    currentRow, currentExpectedResult, netkh, ykh = self.countExcitationAndOutputOfNeurons(currentRow)

                    numberOfLastLayer = self.numberOfHiddenLayers

                    # Sygnał błędu δ dla warstwy wyjściowej
                    delkh = {self.numberOfHiddenLayers : ((currentExpectedResult - ykh[numberOfLastLayer]) * sigmoid_der(netkh[numberOfLastLayer]))}

                    reverseList = list(range(self.numberOfHiddenLayers))
                    reverseList.reverse()

                    # Sygnał błędu δ dla warstw ukrytych. Wsteczna propagacja błędów              
                    for j in reverseList:
                        delkh[j] = sigmoid_der(netkh[j]) * np.dot(self.weights[j+1], delkh[j+1])

                    # Modyfikacja wagi warstwy wyjściowej:
                    self.weights[numberOfLastLayer] += self.eta * delkh[numberOfLastLayer] * ykh[numberOfLastLayer-1]

                    # Modyfikacja wagi warstw ukrytych:
                    for j in reverseList[:-1]:
                        self.weights[j] += self.eta * delkh[j] * ykh[j-1]
                    self.weights[0] += self.eta * delkh[0] * currentRow

                    roundedOutputValue = round(ykh[numberOfLastLayer])

                    if roundedOutputValue == currentExpectedResult:
                        numberOfCorrectlyQualified += 1

                    MSE +=  0.5*(currentExpectedResult - roundedOutputValue)**2

                correctlyQualified = numberOfCorrectlyQualified / len(orderOfRows)
                vecOfCorrectlyQualified.append(correctlyQualified)
                vecOfMSE.append(MSE)
                print("Po {0}. przejsciu calego zbioru treningowego % poprawnie zakwalifikowanych = {1:.5f}.".format(numberOfCurrentEpoch,correctlyQualified*100))

            print("\nProba {0}. zakonczona!\nPo {1}. przejsciu calego zbioru treningowego % poprawnie zakwalifikowanych = {2:.5f}.\n".format(numberOfAttempt+1, numberOfCurrentEpoch, correctlyQualified*100))
            self.weightsOfNeuralNetwork.append((self.weights, correctlyQualified, vecOfMSE, vecOfCorrectlyQualified))

        self.weightsOfNeuralNetwork = sorted(self.weightsOfNeuralNetwork, key=lambda weights: weights[1]) 
        self.weights = self.weightsOfNeuralNetwork[-1][0]
        print("\n\nUczenie calkowicie zakonczone!\nPo {0}. probach najwyzsza poprawnosc wyniosla {1:.5f}.\nZostala zapamietana wartosc wag odpowiadajaca probie o najwyzszej poprawnosci.".format(numberOfAttempt+1, self.weightsOfNeuralNetwork[-1][1] * 100))

        # Rysowanie wykresow
        fig = plt.figure()

        ax1 = fig.add_subplot(121)
        ax1.plot(self.weightsOfNeuralNetwork[-1][2])
        ax1.set_title('MSE')
        ax1.set_ylabel('wartosc MSE')
        ax1.set_xlabel('numer epoki')

        ax2 = fig.add_subplot(122)
        ax2.plot(self.weightsOfNeuralNetwork[-1][3])
        ax2.set_title('% dobrze zakwalifikowanych wierszy')
        ax2.set_ylabel('% poprawnosci')
        ax2.set_xlabel('numer epoki')

        plt.show()

    def countExcitationAndOutputOfNeurons(self, currentRow):
        """
        Oblicza wartosci pobudzen oraz wyjsc wszystkich neuronow w sieci.
        """
        currentExpectedResult = currentRow[0]
        currentRow = np.delete(currentRow, 0, 0)
                
        # Obliczenie wyjsc neuronow na wszystkich warstwach
        netkh = {} # oznacza wejscie na kazda warstwe
        ykh = {} # oznacza wyjscie z kazdej warstwy 

        for j in range(self.numberOfHiddenLayers+1):
            if j == 0:
                netkh[j] = np.dot(self.weights[j], currentRow)
            else: 
                netkh[j] = np.dot(self.weights[j], ykh[j-1])
            ykh[j] = expit(netkh[j]) 

        return currentRow, currentExpectedResult, netkh, ykh

    def countErrorsInZeroEpoch(self):
        """
        Obliczanie bledu MSE oraz % poprawnie zakwalifikowanych dla poczatkowej wartosci wag dla calego zbioru treningowego.
        """
        MSE = 0.0
        numberOfCorrectlyQualified = 0

        for i in range(np.size(self.trainingData,0)):
            currentRow = self.trainingData[i]              

            # Wartosci pobudzen oraz wyjsc wszystkich neuronow w sieci
            currentRow, currentExpectedResult, netkh, ykh = self.countExcitationAndOutputOfNeurons(currentRow)

            roundedOutputValue = round(ykh[self.numberOfHiddenLayers])

            if roundedOutputValue == currentExpectedResult:
                numberOfCorrectlyQualified += 1

            MSE +=  0.5*(currentExpectedResult - roundedOutputValue)**2

        correctlyQualified = numberOfCorrectlyQualified / np.size(self.trainingData,0)

        return correctlyQualified, MSE

    def testNeuralNetwork(self):
        """
        Sprawdza dzialanie sieci neuronowej po nauczeniu. Sprawdzenie odbywa się na danych testowych.
        """
        testData = np.genfromtxt('testData.csv', delimiter=',')
        
        # Czulosc = TP / TP+FN
        TP = 0
        FN = 0
        # Specyficznosc = TN / FP+TN
        TN = 0
        FP = 0

        numberOfCorrectlyQualified = 0

        for i in range(np.size(testData,0)):
            currentRow = testData[i]

            # Wartosci pobudzen oraz wyjsc wszystkich neuronow w sieci
            currentRow, currentExpectedResult, netkh, ykh = self.countExcitationAndOutputOfNeurons(currentRow)

            roundedOutputValue = round(ykh[self.numberOfHiddenLayers])

            if roundedOutputValue == currentExpectedResult:
                numberOfCorrectlyQualified += 1
            if (roundedOutputValue == 1) and (currentExpectedResult == 1):
                TP += 1
            if (roundedOutputValue == 0) and (currentExpectedResult == 1):
                FN += 1
            if (roundedOutputValue == 1) and (currentExpectedResult == 0):
                FP += 1
            if (roundedOutputValue == 0) and (currentExpectedResult == 0):
                TN += 1

        sensitivity = TP / (TP+FN)
        specificity = TN / (FP+TN)
        correctlyQualified = numberOfCorrectlyQualified / np.size(testData,0)
        print("\n\n\nTestowanie ostatecznych wag na zbiorze testowym. Po przejsciu calego zbioru testowego:")
        print("% poprawnie zakwalifikowanych = {0:.5f}.".format(correctlyQualified*100))
        print("Czulosc = {0:.5f}.\nSpecyficznosc = {1:.5f}.\n\n".format(sensitivity,specificity))


def main():
    neuralNetwork = NeuralNetwork(numberOfAttempts = 15, maxNumberOfEpochs = 50, eta = 0.05, numberOfLayers = 3)
    neuralNetwork.teachNeuralNetwork()
    neuralNetwork.testNeuralNetwork()

if __name__ == "__main__":
    main()
