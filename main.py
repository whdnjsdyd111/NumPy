import numpy as np
import os
import matplotlib.pyplot as plt

filepath = os.getcwd()
filename = os.path.join(filepath, "who_covid_19_sit_rep_time_series.csv")

# skip_header 와 usecols 로 데이터 파일을 각 변수에 추가한다.
# 첫 번째 행에서 3-7 열의 날짜만 읽는다.
datas = np.genfromtxt(filename, dtype=np.unicode_, delimiter=",",
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

plt.plot(datas, nbcases.T, '--')
plt.xticks(selected_dates, datas[selected_dates])
plt.title("COVID-19 cumulative cases")
plt.show()
