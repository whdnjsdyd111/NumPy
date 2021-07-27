import csv

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import os

A_M = np.log(2) / 2
B_M = np.log(2250) - A_M * 1971
Moores_law = lambda year: np.exp(B_M) * np.exp(A_M * year)

ML_1971 = Moores_law(1971)
ML_1973 = Moores_law(1973)
print("In 1973, G. Moore expects {:.0f} transistors on Intels chips".format(ML_1973))
print("This is x{:.2f} more transistors than 1971".format(ML_1973 / ML_1971))

data = np.loadtxt('transistor_data.csv', delimiter=',',
                  usecols=[1, 2], skiprows=1)

year = data[:, 1]
transistor_count = data[:, 0]

print("year:\t\t", year[:10])
print("trans. cnt:\t", transistor_count[:10])

yi = np.log(transistor_count)
Z = year[:, np.newaxis] ** [1, 0]
model = sm.OLS(yi, Z)

results = model.fit()
print(results.summary())
