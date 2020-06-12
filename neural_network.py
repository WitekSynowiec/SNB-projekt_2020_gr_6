import numpy as np
import pandas as pd
from scipy.special import expit, logit
import time


# Funkcja pochodna sigmoidy
def sigmoid_der(xx):
    return expit(-xx) * (1 - expit(-xx))


# Sieć neuronowa
def neuralNetwork(data, wih, whj, eta):
    # Losowy wektor uczący
    vec = data[np.random.randint(0, np.size(data, 0))]
    # zapisujemy wartość oczekiwaną tzn 0 lub 1
    vec0 = vec[0]
    vec = np.delete(vec, 0, 0)

    netkh = np.dot(whj, vec)

    # Stan wyjść neuronów warstwy ukrytej
    ykh = expit(netkh)

    # Pobudzenia neuronów warstwy wyjściowej
    netki = np.dot(wih, ykh)

    # Stan wyjść neuronów warstwy wyjściowej
    yki = expit(netki)

    # Sygnał błędu δ dla warstwy wyjściowej
    delki = (vec0 - yki) * sigmoid_der(netki)

    # Sygnał błędu δ dla warstwy wyjściowej. Wsteczna propagacja błędów
    delkh = sigmoid_der(netkh) * np.dot(wih, delki)

    # Modyfikacja wagi warstwy wyjściowej:
    wih = wih + eta * delki * ykh

    # Modyfikacja wagi warstwy ukrytej:
    whj = whj + eta * delkh * vec

    return [data, wih, whj, eta, ykh, yki]




# Wczytywanie danych z pliku data i zapisanie do struktury data
data = np.genfromtxt('data.csv', delimiter=',')
data = np.delete(data, 0, 0)

# Ustawienie małuch, losowych wag z przedziału [-1,1]
# Wagi początkowe warstwy ukrytej (czyli 15*15 wag)
whj = 2 * np.random.rand(np.size(data, 1) - 1, np.size(data, 1) - 1) - 1
# Wagi początkowe warstwy wyjściowej (15 wag)
wih = 2 * np.random.random(np.size(data, 1) - 1) - 1

eta = 0.05
ykh = 0
yki = 0

# TODO: testy sieci, czy dobrze wyrzuca wagi
for x in range(400000):
    [data, wih, whj, eta, ykh, yki] = neuralNetwork(data, wih, whj, eta)

c=0
for ii in range(100):
    vec = data[np.random.randint(0, np.size(data, 0))]
    vec0 = vec[0]
    vec = np.delete(vec, 0, 0)

    print("Wartość oczekiwana: ")
    print(vec0)

    ykhh = expit(np.dot(whj, vec))

    ykii = expit(np.dot(ykhh, wih))
    print("Wartość uzyskana: ")
    print(ykii)

    print("Werdykt: ")
    print(round(ykii))
    if(round(ykii)==vec0):
        print("Zgadza się.")
        c=c+1
        print(ii)
        print(c)
    else:
        print("Nie zgadza się.")
        print(ii)
        print(c)

print(c)