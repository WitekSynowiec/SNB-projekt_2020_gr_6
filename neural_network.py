import numpy as np
import pandas as pd



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


data = np.genfromtxt('data.csv', delimiter=',')
data = np.delete(data, 0, 0)
print(pd.DataFrame(data))

