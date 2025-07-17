import pandas as pd

data="Datasets/daily-min-temperatures.csv"
df=pd.read_csv(data,parse_dates=['Date'],index_col="Date")
series=df['Temp']

window_size=7
moving_avg=series.rolling(window=window_size).mean()

forecast=moving_avg.iloc[-1]

print(f"Forecasted next value using {window_size}-day moving average:{forecast:.3f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(series,label="Daily min Temperature")
plt.plot(moving_avg,label=f'{window_size}-day moving Average',linewidth=2)
plt.xlabel("Date")
plt.ylabel("Temperature(C)")
plt.title("Moving average forecasting on daily min Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

