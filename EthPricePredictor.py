import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import  Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import requests
from sklearn.metrics import mean_squared_error

crypto = "ETH"
euros = "EUR"
api_key = "f29f27ae2e16d62df1ecb591ced19d4d1524423b87b73b18efaa66057fe1aeee"
start_date = "2019-01-01"#cryptocompare limit LOL 
end_date = "2023-04-06"
PREDICTION_DAYS= 30
FUTURE_DAYS = 15


start_timestamp = int(pd.Timestamp(start_date).timestamp())
end_timestamp = int(pd.Timestamp(end_date).timestamp())

# Calculate number of days between start and end dates
num_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days

url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={crypto}&tsym={euros}&limit={num_days}&toTs={end_timestamp}&api_key={api_key}"
response = requests.get(url)
data = response.json()["Data"]["Data"]
df = pd.DataFrame(data)
df["time"] = df["time"].apply(lambda x: dt.date.fromtimestamp(x))

#Prepare the data for trainning

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df["close"].values.reshape(-1,1))


x_train, y_train = [], []

for x in range(PREDICTION_DAYS, len(scaled_data)-FUTURE_DAYS):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x, 0])
    y_train.append(scaled_data[x+FUTURE_DAYS,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences= True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss = "mean_squared_error")
model.fit(x_train, y_train, epochs= 20, batch_size=30)


#Testing Data

test_start = dt.datetime(2022,1,1)
test_end = dt.datetime.now()
num_days_test_data = (pd.Timestamp(test_end) - pd.Timestamp(test_start)).days
end_timestamp_test_data = int(pd.Timestamp(test_end).timestamp())


urlTest_data = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={crypto}&tsym={euros}&limit={num_days_test_data}&toTs={end_timestamp_test_data}&api_key={api_key}"
response_test_data = requests.get(urlTest_data)
data_test = response_test_data.json()["Data"]["Data"]
df_test = pd.DataFrame(data_test)
df_test["time"] = df_test["time"].apply(lambda x: dt.date.fromtimestamp(x))

#Prepare test data
actual_prices = df_test["close"].values

total_data_set = pd.concat((df["close"],df_test["close"]),axis=0)

model_inputs = total_data_set[len(total_data_set)-len(df_test)-PREDICTION_DAYS:].values
model_inputs= model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x-PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, color = "black", label = "Actual Prices")
plt.plot(prediction_prices, color = "red", label= "Predicted Prices")
plt.title(f"{crypto} Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc = "upper left")
plt.show()

#Accuracy

mse = mean_squared_error(actual_prices, prediction_prices)

print("Mean Squared Error:", mse)

#Predicting the future

real_data = [model_inputs[len(model_inputs)+1-PREDICTION_DAYS:len(model_inputs)+1,0]]
real_data= np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0],real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction)