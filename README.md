# Utilizing Recurrent Neural Networks via LSTM to Forecast Stock Prices

![LSTM](https://github.com/NavarroAlexKU/Forecasting-Stock-Price-Using-Long-Short-Term-Memory-LSTM-/blob/main/LSTM%20Arch.png?raw=true)

This project implements a Long Short-Term Memory (LSTM) network using TensorFlow and Keras to predict stock prices. The project involves data preprocessing, building the LSTM model, training it on a dataset, and making predictions on new data.

## Table of Contents
- [Stock Price Prediction Using Long Short-Term Memory (LSTM)](#stock-price-prediction-using-long-short-term-memory-lstm)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Data Preprocessing](#data-preprocessing)
    - [Importing the Libraries](#importing-the-libraries)
  - [Importing the Training Set:](#importing-the-training-set)
  - [Feature Scaling:](#feature-scaling)
  - [Creating A Data Structure with TimeSteps:](#creating-a-data-structure-with-timesteps)
  - [Reshaping Data:](#reshaping-data)
  - [Buildling and Training the LSTM:](#buildling-and-training-the-lstm)
  - [Adding the Output Layer:](#adding-the-output-layer)
  - [Compiling the LSTM](#compiling-the-lstm)
  - [Training the LSTM](#training-the-lstm)
  - [Making Predictions and Visualizing the Results](#making-predictions-and-visualizing-the-results)

## Project Overview

This project demonstrates the implementation of a Long Short-Term Memory (LSTM) network to predict stock prices. The dataset includes historical stock prices, which are used to train the model. The LSTM is built, trained, and evaluated on these datasets. Finally, the trained model is used to make predictions on new data.

## Data Preprocessing

### Importing the Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Configure TensorFlow to use 16 CPU cores
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
```

## Importing the Training Set:
```
# Load the training dataset
dataset_train = pd.read_csv(r'Google_Stock_Price_Train.csv')

# Select the column containing the training data
training_set = dataset_train.iloc[:, 1:2].values
```

## Feature Scaling:
In this section, we scale the stock price data to ensure that it falls within a specific range, which helps improve the performance and convergence of the LSTM model. Feature scaling is a common preprocessing step in machine learning that standardizes the range of independent variables.
```
# Scale the data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
```

## Creating A Data Structure with TimeSteps:
In this section, we create a data structure that incorporates the concept of timesteps, which is essential for training the LSTM model. The idea is to use historical stock price data to predict future prices. Specifically, for each day, we use the stock prices from the previous 60 days to predict the stock price of the next day.
```
# Initialize the lists to hold our training data
X_train = []
y_train = []

# Loop over the dataset to create the sequences
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# Convert the lists to numpy arrays to use them in the model
X_train, y_train = np.array(X_train), np.array(y_train)
```

## Reshaping Data:
In this section, we reshape the training data to be in the appropriate format required by the LSTM model. The LSTM expects the input data to be in three-dimensional shape: [samples, timesteps, features].
```
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```

## Buildling and Training the LSTM:
In this section, we initialize the LSTM model using the Keras Sequential API. The Sequential model is a linear stack of layers that allows us to build a neural network layer by layer.
```
# Initializing the LSTM
regressor = Sequential()

# Add the Input layer to the regressor
regressor.add(Input(shape=(X_train.shape[1], 1)))
```
Adding the LSTM Layers and Dropout Registration:
Now we build the LSTM network by adding multiple LSTM layers and incorporating dropout regularisation to prevent overfitting. Each LSTM layer captures temporal dependencies in the data, and dropout regularisation helps improve the model's generalization by randomly setting a fraction of input units to 0 during training.
```
# Add the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Add the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Add the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Add the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
```

## Adding the Output Layer:
# Add the output Dense layer to the regressor
regressor.add(Dense(units=1))
```
# Add the output Dense layer to the regressor
regressor.add(Dense(units=1))
```

## Compiling the LSTM
In this section, we compile the LSTM model by specifying the optimizer and the loss function. Compiling the model is a necessary step before training, as it configures the model for training.
```
# Compile the regressor
regressor.compile(optimizer='adam', loss='mean_squared_error')
```

## Training the LSTM
Now we finally train our LSTM model setting epochs to 100.
```
# Train the regressor on the training data
regressor.fit(X_train, y_train, epochs=100, batch_size=32)
```

## Making Predictions and Visualizing the Results
Now we load the test dataset, preprocess the data, make predictions using the trained LSTM model, and visualize the results.
```
# Load the test dataset
dataset_test = pd.read_csv(r'Google_Stock_Price_Test.csv')

# Extract the real stock prices from the test dataset
real_stock_price = dataset_test.iloc[:, 1:2].values

# Concatenate the training and test data
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

# Get the inputs for the model
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

# Reshape the inputs
inputs = inputs.reshape(-1, 1)

# Scale the inputs
inputs = sc.transform(inputs)

# Predict the stock prices
predicted_stock_price = regressor.predict(inputs)

# Inverse scale the predictions
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plot the real and predicted stock prices
plt.figure(figsize=(14, 7))
plt.plot(real_stock_price, color='blue', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```
![prediction vs. actual](https://raw.githubusercontent.com/NavarroAlexKU/Forecasting-Stock-Price-Using-Long-Short-Term-Memory-LSTM-/main/predicted%20vs.%20actual.jfif)

### Plot Description:

- Real Stock Price:
  - The blue line represents the actual stock prices from the test dataset.
- Predicted Stock Price:
  - The red line shows the stock prices predicted by the LSTM model.
    
## Observations:

- Trend Matching:
    - The predicted prices closely follow the trend of the real stock prices, indicating that the model has captured the general movement of the stock.
- Fluctuations:
    - There are minor discrepancies between the predicted and real prices, especially around points of sharp movement. These could be due to the model's limitations or the inherent volatility in stock prices.
- Model Performance:
    - The overall alignment of the predicted and real prices suggests that the LSTM model has effectively learned from the training data, though there is room for improvement, particularly in capturing sudden changes.
