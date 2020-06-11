import numpy as np
import pandas as pd


# Funkcja sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Wczytywanie danych z pliku data i zapisanie do struktury data
data = np.genfromtxt('data.csv', delimiter=',')
data = np.delete(data, 0, 0)
print(pd.DataFrame(data))

# Ustawienie małuch, losowych wag z przedziału [-1,1]
np.random.seed(1)
# Wagi początkowe warstwy wejściowej
wih = 2 * np.random.random(np.size(data,1)) - 1
# Wagi początkowe warstwy ukrytej
whj = 2 * np.random.random(np.size(data,1)) - 1

# print(pd.DataFrame(wih))

# Losowy wektor uczący
vec = data[np.random.randint(0, np.size(data,0))]

print(vec)