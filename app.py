import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


model = load_model("Stock Prediction using LSTM.keras")

st.header('Stock Prediction Predictor')

stock = st.text_input("Enter Stock Symbol", 'GOOG')
start = st.text_input("Enter start_date", '1995-01-01')
end = st.text_input("Enter end date", '2025-10-20')

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

X_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
X_test = pd.DataFrame(data.Close[int(len(data)*0.80) : len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = X_train.tail(100)
X_test = pd.concat([past_100_days, X_test], ignore_index=True)
X_test_scale = scaler.fit_transform(X_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(6, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(6, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig2)

st.subheader('Price vs MA50 vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(6, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(ma_200_days, 'y')
plt.plot(data.Close, 'g')
st.pyplot(fig3)

x = []
y = []

for i in range(100, X_test_scale.shape[0]):
    x.append(X_test_scale[i-100: i])
    y.append(X_test_scale[i, 0])
    
x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_
predict = predict*scale
y = y*scale


st.subheader('Price vs Predicted Price')
fig4 = plt.figure(figsize=(6, 6))
plt.plot(predict, 'r', label = 'Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig4)