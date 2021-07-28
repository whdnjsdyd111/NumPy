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

AB = results.params
A = AB[0]
B = AB[1]

print("Rate of semiconductors added on a chip every 2 years:")
print(
    "\tx{:.2f} +/- {:.2f} semiconductors per chip".format(
        np.exp(A * 2), 2 * A * np.exp(2 * A) * 0.006
    )
)

transistor_count_predicted = np.exp(B) * np.exp(A * year)
transistor_Moores_law = Moores_law(year)
plt.style.use("fivethirtyeight")
plt.semilogy(year, transistor_count, "s", label = "MOS transistor count")
plt.semilogy(year, transistor_count_predicted, label = "linear regression")


plt.plot(year, transistor_Moores_law, label = "Moore's Law")
plt.title(
    "MOS transistor count per microprocessor\n"
    + "every two years \n"
    + "Transistor count was x{:.2f} higher".format(np.exp(A * 2))
)
plt.xlabel("year introduced")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.ylabel("# of transistors\nper microprocessor")
plt.show()