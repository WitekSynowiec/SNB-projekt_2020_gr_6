import numpy as np
import pandas as pd


# Funkcja sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Funkcja pochodna
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Sieć neuronowa
def neuralNetwork(data, wih, whj, eta):
    # Losowy wektor uczący
    vec = data[np.random.randint(0, np.size(data, 0))]

    # Pobudzenia neuronów warstwy ukrytej
    netkh = np.multiply(whj, vec)
    # print(pd.DataFrame(netkh))

    # Stan wyjść neuronów warstwy ukrytej
    ykh = sigmoid(netkh)

    # Pobudzenia neuronów warstwy wyjściowej
    netki = np.multiply(wih, ykh)

    # Stan wyjść neuronów warstwy wyjściowej
    yki = sigmoid(netki)

    # Sygnał błędu δ dla warstwy wyjściowej
    delki = (vec[0] - yki) * sigmoid_der(netki)

    # Sygnał błędu δ dla warstwy wyjściowej. Wsteczna propagacja błędów
    delkh = sigmoid_der(netkh) * np.multiply(wih, delki)

    # Modyfikacja wagi warstwy wyjściowej:
    wih = wih + eta * delki * ykh

    # Modyfikacja wagi warstwy ukrytej:
    whj = whj + eta * delkh * vec

    return [data, wih, whj, eta]


# Wczytywanie danych z pliku data i zapisanie do struktury data
data = np.genfromtxt('data.csv', delimiter=',')
data = np.delete(data, 0, 0)
# print(pd.DataFrame(data))

# Ustawienie małuch, losowych wag z przedziału [-1,1]
np.random.seed(1)
# Wagi początkowe warstwy wejściowej
wih = 2 * np.random.random(np.size(data, 1)) - 1
# Wagi początkowe warstwy ukrytej
whj = 2 * np.random.random(np.size(data, 1)) - 1

eta = 0.05


# TODO: testy sieci, czy dobrze wyrzuca wagi
for x in range(100000):
    [data, wih, whj, eta] = neuralNetwork(data, wih, whj, eta)


