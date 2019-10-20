import numpy as np
import pandas as pd
from pandas import datetime
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pylab as plt

def parser(x):
    return datetime.strptime(x, "%Y-%m-%d")


sales = pd.read_csv('Book1.csv', index_col=0, parse_dates=[0], date_parser=parser)
rollmean = sales.rolling(window=12).mean()
rollstd= sales.rolling(window=12).std()
# sales.plot()
# plt.show()
# orig = plt.plot(sales, color='blue', label='Original')
# mean= plt.plot(rollmean, color='red', label='Mean')
# std = plt.plot(rollstd, color='black', label='Rolling Standard Deviation')
# plt.show()

from statsmodels.tsa.stattools import adfuller

indexDataset_logScale = np.log(sales+1)
movingAverage = indexDataset_logScale.rolling(window=12).mean()
movingSTD= indexDataset_logScale.rolling(window=12).std()
#
dataset_logScaleMinusMovingAverage = indexDataset_logScale - movingAverage
dataset_logScaleMinusMovingAverage.head(12)

dataset_logScaleMinusMovingAverage.dropna(inplace=True)
# print(dataset_logScaleMinusMovingAverage.head(5))
#
#
def test_stationarity(timeseries):
    movingAverage=timeseries.rolling(window=12).mean()
    movingSTD=timeseries.rolling(window=12).std()
    #Plot Statics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling STD')

    print("Result of Dickeyfuller test")
    dftest = adfuller(timeseries.Sales, autolag='AIC')

    dfouput = pd.Series(dftest[0:4], index=['Test Statics', 'p-value', '#Lags Used', 'Number of Observation Used'])
    for key, value in dftest[4].items():
        dfouput['Critical Value (%s)' % key] = value
    print(dfouput)
    plt.show()
#
#
exponentialDecayWeightedAverage = indexDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
#
dataset_logScaleMinusMovingExponentialDecayAverage = indexDataset_logScale - exponentialDecayWeightedAverage

dataset_logDiffShifting = indexDataset_logScale - indexDataset_logScale.shift()
dataset_logDiffShifting.dropna(inplace=True)

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(dataset_logDiffShifting, nlags=20)
lag_pacf = pacf(dataset_logDiffShifting, nlags=20, method='ols')

# #plt ACF
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0, linestyle='--', color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(dataset_logDiffShifting)), linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(dataset_logDiffShifting)), linestyle='--',color='grey')
#
# #PACF
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0, linestyle='--', color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(dataset_logDiffShifting)), linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(dataset_logDiffShifting)), linestyle='--',color='gray')
# plt.title('Autocorrelation')
#
# plt.title('partila')
# plt.tight_layout()
# # plt.show()


from statsmodels.tsa.arima_model import ARIMA



#AR MODEL
# model = ARIMA(indexDataset_logScale, order=(2,1,0))
# resultAR = model.fit(disp=-1)
# plt.plot(resultAR.fittedvalues, color='red')
# plt.title("RSS : %.4f"% sum((resultAR.fittedvalues-dataset_logDiffShifting.Sales)**2))
# plt.show()
#
# #ARIMA MODEL
model = ARIMA(indexDataset_logScale, order=(2,1,0))
result_ARIMA = model.fit(disp=-1)
# plt.plot(dataset_logDiffShifting)
# plt.plot(result_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((result_ARIMA.fittedvalues-dataset_logDiffShifting.Sales)**2))
# plt.show()

prediction_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)

prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()

prediction_ARIMA_log = pd.Series(indexDataset_logScale.Sales.ix[0], index=indexDataset_logScale.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum, fill_value=0)

predictions_ARIMA = np.exp(prediction_ARIMA_log)

result_ARIMA.plot_predict(1,220)
x=result_ARIMA.forecast(steps=120)
plt.show()