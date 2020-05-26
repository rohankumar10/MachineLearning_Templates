# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Aaple_Stock_Price_Train.csv')
# We want to create a numpy arrary not a vector hence 1:2
training_set = dataset_train.iloc[:, 1:2].values


# Feature scaling to optimize the training set
# Apply normalisation in RNN 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 output
# 60 T before T and predict the output at T+1
X_train = []
y_train = []
for i in range(120, 4779):
    X_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding first LSTM layer and some dropout Dropout regularisation
regressor.add(LSTM(units=100, return_sequences=True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding second LSTM layer and some dropout Dropout regularisation
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding third LSTM layer and some dropout Dropout regularisation
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding fourth LSTM layer and some dropout Dropout regularisation
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding fifth LSTM layer and some dropout Dropout regularisation
regressor.add(LSTM(units=100))
regressor.add(Dropout(0.2))

# Adding the Output Layer
regressor.add(Dense(units=1))

# Compiling the RNN
# Because we're doing regression hence mean_squared_error
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 150, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Aaple_Stock_Price_Test.csv')
# We want to create a numpy arrary not a vector hence 1:2
real_stock_price = dataset_test.iloc[:, 1:2].values
# Getting the predicted stock price of Jan 2020
# We have to concatenate both sets as the 60 T will have data from Dec 2019 and January 2020
# We shouldn't concatenate the train and test set as we would have to apply feature scaling
# to the concatenated result which would change the test prices
# Hence we will concatenate the dataset and then scale them
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(120, 141):
    X_test.append(inputs[i-120:i, 0])

X_test = np.array(X_test)
# 3D format
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
# Inverse the scaling
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color='Red', label='Real Aaple Stock Price')
plt.plot(predicted_stock_price, color='Blue', label='Predicted Aaple Stock Price')
plt.title('Aaple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Aaple Stock Price')
plt.legend()
plt.show()




