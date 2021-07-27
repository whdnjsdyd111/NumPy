import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import ma

filepath = os.getcwd()
filename = os.path.join(filepath, "who_covid_19_sit_rep_time_series.csv")

# skip_header 와 usecols 로 데이터 파일을 각 변수에 추가한다.
# 첫 번째 행에서 3-7 열의 날짜만 읽는다.
dates = np.genfromtxt(filename, dtype=np.unicode_, delimiter=",",
                      max_rows=1, usecols=range(3, 17),
                      encoding="utf-8-sig")

# 처음 두 컬럼으로 부터 위치 이름을 읽는다.
# 처음 7 행은 생략한다.
locations = np.genfromtxt(filename, dtype=np.unicode_, delimiter=",",
                          skip_header=7, usecols=(0, 1),
                          encoding="utf-8-sig")

# 처음 14일 동안의 숫자 데이터를 읽는다.
nbcases = np.genfromtxt(filename, dtype=np.int_, delimiter=",",
                        skip_header=7, usecols=range(3, 17),
                        encoding="utf-8-sig")

selected_dates = [0, 3, 11, 13]

nbcases_ma = ma.masked_values(nbcases, -1)

china_masked = nbcases_ma[locations[:, 1] == 'China'].sum(axis=0)

china_mask = ((locations[:, 1] == 'China') &
              (locations[:, 0] != 'Hong Kong') &
              (locations[:, 0] != 'Taiwan') &
              (locations[:, 0] != 'Macau') &
              (locations[:, 0] != 'Unspecified*'))

china_total = nbcases_ma[china_mask].sum(axis=0)

invalid = china_total[china_total.mask]
valid = china_total[~china_total.mask]

t = np.arange(len(china_total))
params = np.polyfit(t[~china_total.mask], valid, 3)
cubic_fit = np.polyval(params, t)
plt.plot(t, china_total, label='Mainland China');

plt.plot(t[china_total.mask], cubic_fit[china_total.mask], '--',
         color='orange', label='Cubic estimate')
plt.plot(7, np.polyval(params, 7), 'r*', label='7 days after start')
plt.xticks([0, 7, 13], dates[[0, 7, 13]])
plt.yticks([0, np.polyval(params, 7), 10000, 17500])
plt.legend()
plt.title("COVID-19 cumulative cases from Jan 21 to Feb 3 2020 - Mainland China\n"
          "Cubic estimate for 7 days after start")
plt.show()
