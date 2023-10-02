# Framework to operate on Web
import streamlit as st

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Statistics, Linear Algebra and Data Manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import pmdarima as pm
from sklearn.model_selection import train_test_split

# API for USD(BRL)
import quandl

# API for BRENT(USD)
import yfinance as yf

st.title("Data Science for Oil Price")

# Date Picker Widget
start_date = st.date_input("Select a start date", pd.to_datetime("2020-04-21")).strftime("%Y-%m-%d")
end_date = st.date_input("Select an end date", pd.to_datetime("2022-06-11")).strftime("%Y-%m-%d")

if  st.button("Process!"):
    # Generate Data
    df_brent = yf.download("BZ=F", start=start_date, end=end_date)
    df_brent = df_brent[["Adj Close"]].rename({"Adj Close": "Value"}, axis=1)
    df_currency = quandl.get("BCB/10813", start_date=start_date, end_date=end_date)

    #https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
    st.write("Brent(USD) and BRL(USD).")
    st.write(df_brent)
    fig, ax1 = plt.subplots(figsize=(15,15))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    for i in [{"df": df_brent, "color": "tab:red", "ax": ax1, "name": "Brent (USD)"},{"df": df_currency, "color": "tab:blue", "ax": ax2, "name": "USD (BRL)"}]:
        i["ax"].set_ylabel(i["name"], color=i["color"])
        i["ax"].plot(i["df"].index, i["df"].Value, color=i["color"])
        i["ax"].tick_params(axis="y", labelcolor=i["color"])
    plt.grid()
    plt.show()
    st.pyplot(fig)

    result = seasonal_decompose(df_brent["Value"], period=12, model="additive")

    st.title("Trend")
    fig,ax = plt.subplots(figsize=(15,15))
    plt.plot(df_brent.index, result.trend)
    plt.show()
    st.pyplot(fig)

    st.title("Seasonal")
    fig,ax = plt.subplots(figsize=(15,15))
    plt.plot(df_brent.index, result.seasonal)
    plt.show()
    st.pyplot(fig)

    st.title("Residuals")
    fig,ax = plt.subplots(figsize=(15,15))
    plt.plot(df_brent.index, result.resid)
    plt.show()
    st.pyplot(fig)

    st.title("Observed")
    fig,ax = plt.subplots(figsize=(15,15))
    plt.plot(df_brent.index, result.observed)
    plt.show()
    st.pyplot(fig)

        
    #https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/
    #https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_acf.html
    st.title("Plots lags on the horizontal and the correlations on vertical axis.")
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14,6), sharex=False, sharey=False)
    ax1 = plot_acf(df_brent["Value"], lags=50, ax=ax1)
    ax2 = plot_pacf(df_brent["Value"], lags=50, ax=ax2)
    plt.show()
    st.pyplot(fig)

    st.title("Dickey-Fueller (Non-seasonal)")
    dftest = adfuller(df_brent["Value"])
    dfoutput = pd.Series(dftest[0:4], index=["Test Statistic","p-value","#Lags Used","Number of Observations Used"])
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)"%key] = value
    st.write(dfoutput)

    st.title("Dickey-Fueller (Seasonal)")
    df_diff = df_brent.diff().diff(12).dropna()
    dftest = adfuller(df_diff["Value"])
    dfoutput = pd.Series(dftest[0:4], index=["Test Statistic","p-value","#Lags Used","Number of Observations Used"])
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)"%key] = value
    st.write(dfoutput)

    #https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
    st.title("Automatically discover the optimal order for an ARIMA model.")
    model = pm.auto_arima(df_brent["Value"], d=1, D=1,
                        seasonal=True, m=12, trend="c", 
                        start_p=0, start_q=0, max_order=6, test="adf", stepwise=True, trace=True)
    st.write(model.summary())

    st.title("Train/Test Database")
    train, test = train_test_split(df_brent, test_size=0.15, shuffle=False)
    fig,ax = plt.subplots(figsize=(15,15))
    train["Value"].plot()
    test["Value"].plot()
    st.pyplot(fig)

    #https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.sarimax.html
    st.title("Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model")
    #model = SARIMAX(train["Value"],order=(2,1,0),seasonal_order=(1,1,1,12))
    model = SARIMAX(df_brent["Value"],order=(1,1,1),seasonal_order=(1,1,1,12))
    results = model.fit()
    st.write(results.summary())
    st.pyplot(results.plot_diagnostics(figsize=(16, 8)))


    #https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.mlemodel.MLEModel.fit.html#statsmodels.tsa.statespace.mlemodel.MLEModel.fit
    #https://machinelearningmastery.com/time-series-forecast-uncertainty-using-confidence-intervals-python/
    st.title("Sample forecast")
    #forecast_object = results.get_forecast(steps=len(test))
    forecast_object = results.get_prediction(start=test.index[0],dynamic=False)
    mean = forecast_object.predicted_mean
    confidence_interval = 0.30
    conf_int = forecast_object.conf_int(alpha = 1.0 - confidence_interval)
    dates = test.index
    fig,ax = plt.subplots(figsize=(15,15))
    plt.plot(df_brent.index, df_brent, label="real")
    plt.plot(dates, mean, label="predicted")
    plt.fill_between(dates, conf_int.iloc[:,0], conf_int.iloc[:,1],alpha=0.20)
    plt.show()
    st.pyplot(fig)

    st.title("Difference between predicted and forecast")
    difference = np.asarray(test["Value"]) - np.asarray(mean)
    fig,ax = plt.subplots(figsize=(15,15))
    plt.scatter(test.index, difference)
    plt.show()
    st.pyplot(fig)

    st.title("Forecast's metrics")
    evaluation_results = pd.DataFrame({"r2_score": r2_score(test["Value"], mean)}, index=[0])
    evaluation_results["mean_absolute_error"] = mean_absolute_error(test["Value"], mean)
    evaluation_results["mean_squared_error"] = mean_squared_error(test["Value"], mean) 
    evaluation_results["mean_squared_error"] = mean_absolute_percentage_error(test["Value"], mean)
    st.write(evaluation_results)

    st.title("Predictions in 120 months")
    delta_months = 30*6
    confidence_interval = 0.30
    dates = pd.date_range(df_brent.index[-1], periods=delta_months, freq="D")
    dates = dates[1:]
    forecast_object = results.get_forecast(steps=len(dates))
    mean = forecast_object.predicted_mean
    conf_int = forecast_object.conf_int(alpha = 1.0 - confidence_interval)
    fig,ax = plt.subplots(figsize=(15,15))
    plt.plot(df_brent.index, df_brent, label="real")
    plt.plot(dates, mean, label="predicted")
    plt.fill_between(dates, conf_int.iloc[:,0], conf_int.iloc[:,1],alpha=0.20)
    st.pyplot(fig)
