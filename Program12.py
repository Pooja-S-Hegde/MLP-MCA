import pandas as pd

data="Datasets/airline-passengers.csv"
df=pd.read_csv(data,parse_dates=['Month'],index_col="Month")
df.index.freq='MS'

from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(df['Passengers'],order=(2,1,2))
model_fit=model.fit()

forecast=model_fit.forecast(steps=1).iloc[0]
print(f"Forecasted next value using ARIMA:{forecast:.2f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(df['Passengers'],label="Original Data")
plt.plot(model_fit.fittedvalues,label='Fitted values',linestyle='--')
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.title("ARIMA forecast on airline passengers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
