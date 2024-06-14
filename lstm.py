#!/usr/bin/env python
# coding: utf-8

# # Recurrent Neural Network

# ## Part 1 - Data Preprocessing

# ### Importing the libraries

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# Configure TensorFlow to use 16 CPU cores
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Input


# ### Importing the training set

# In[16]:


# Load the training dataset
dataset_train = pd.read_csv(r'Google_Stock_Price_Train.csv')  # Load the training data from a CSV file

# Select the column containing the training data
# We use iloc to select all rows and the second column (index 1)
# .values is used to convert the DataFrame column into a numpy array
training_set = dataset_train.iloc[:, 1:2].values  # Extract the relevant column (stock price) for training


# ### Feature Scaling

# In[17]:


# scale the data:
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)


# ### Creating a data structure with 60 timesteps and 1 output

# In[18]:


# Creating a data structure with 60 timesteps and 1 output
# This means that for each element of X_train, we have 60 previous days' stock prices
# and for each element of y_train, we have the stock price of the next day.

# Initialize the lists to hold our training data
X_train = []  # List to hold the sequences of 60 previous days' stock prices
y_train = []  # List to hold the stock price of the next day

# Loop over the dataset to create the sequences
for i in range(60, 1258):
    # Append the past 60 days' stock prices to X_train
    X_train.append(training_set_scaled[i-60:i, 0])
    # Append the current day's stock price to y_train
    y_train.append(training_set_scaled[i, 0])

# Convert the lists to numpy arrays to use them in the model
X_train, y_train = np.array(X_train), np.array(y_train)


# ### Reshaping

# In[19]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# ## Part 2 - Building and Training the RNN

# ### Initialising the RNN

# In[20]:


# Initializing the RNN:
regressor = Sequential()

# Add the Input layer to the regressor
# 'shape' defines the shape of the input data (timesteps, features)
regressor.add(Input(shape=(X_train.shape[1], 1)))


# ### Adding the first LSTM layer and some Dropout regularisation

# In[22]:


# Add the first LSTM layer to the regressor
# 'units' specifies the number of neurons in the LSTM layer
# 'return_sequences=True' means the layer will return the full sequence of outputs for the next LSTM layer
regressor.add(
    LSTM(
        units=50,
        return_sequences=True
    )
)

# Add a Dropout layer to the regressor
# '0.2' means 20% of the neurons will be randomly dropped during training to prevent overfitting
regressor.add(
    Dropout(0.2)
)


# ### Adding a second LSTM layer and some Dropout regularisation

# In[23]:


# Add the second LSTM layer to the regressor
# 'units' specifies the number of neurons in the LSTM layer
# 'return_sequences=True' means the layer will return the full sequence of outputs to the next LSTM layer
regressor.add(
    LSTM(
        units=50,
        return_sequences=True
    )
)

# Add another Dropout layer to the regressor
# '0.2' means 20% of the neurons will be randomly dropped during training to prevent overfitting
regressor.add(
    Dropout(0.2)
)


# ### Adding a third LSTM layer and some Dropout regularisation

# In[24]:


# Add the third LSTM layer to the regressor
# 'units' specifies the number of neurons in the LSTM layer
# 'return_sequences=True' means the layer will return the full sequence of outputs to the next LSTM layer
regressor.add(
    LSTM(
        units=50,
        return_sequences=True
    )
)

# Add another Dropout layer to the regressor
# '0.2' means 20% of the neurons will be randomly dropped during training to prevent overfitting
regressor.add(
    Dropout(0.2)
)


# ### Adding a fourth LSTM layer and some Dropout regularisation

# In[25]:


# Add the final LSTM layer to the regressor
# 'units' specifies the number of neurons in the LSTM layer
# Since this is the last LSTM layer, 'return_sequences' is not needed
regressor.add(
    LSTM(
        units=50
    )
)

# Add another Dropout layer to the regressor
# '0.2' means 20% of the neurons will be randomly dropped during training to prevent overfitting
regressor.add(
    Dropout(0.2)
)


# ### Adding the output layer

# In[26]:


# Add the output Dense layer to the regressor
# 'units=1' means this layer has a single neuron, which is typical for regression tasks
# This layer will output the predicted value
regressor.add(
    Dense(
        units=1
    )
)


# ### Compiling the RNN

# In[27]:


# Compile the regressor
# 'optimizer' specifies the algorithm to use for minimizing the loss function
# 'adam' is an adaptive learning rate optimization algorithm that's popular for deep learning
# 'loss' specifies the loss function to be minimized, 'mean_squared_error' is common for regression tasks
regressor.compile(
    optimizer='adam',
    loss='mean_squared_error'
)


# ### Fitting the RNN to the Training set

# In[28]:


# Train the regressor (neural network) on the training data
# X_train: Input features for the training set
# y_train: Target values for the training set
# epochs: Number of times the entire training set is passed through the network
# batch_size: Number of samples processed before the model is updated
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


# ## Part 3 - Making the predictions and visualising the results

# ### Getting the real stock price of 2017

# In[14]:


# Load the test dataset from a CSV file
# The r before the file path indicates a raw string, which means backslashes are treated as literal characters
dataset_test = pd.read_csv(r'Google_Stock_Price_Test.csv')

# Extract the real stock prices from the test dataset
# iloc[:, 1:2] selects all rows (:) and the second column (1:2) from the dataset
# .values converts the selected data into a numpy array
real_stock_price = dataset_test.iloc[:, 1:2].values


# ### Getting the predicted stock price of 2017

# In[29]:


# Getting the predicted stock prices of 2017
# Concatenate the 'Open' column of the training and test datasets along the rows (axis=0)
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

# Get the inputs for the model from the concatenated dataset
# Select the values from the concatenated dataset starting from (total length - length of test set - 60) to the end
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

# Reshape the inputs to have one column (required for scaling)
inputs = inputs.reshape(-1, 1)

# Scale the inputs using the previously fitted scaler (sc)
inputs = sc.transform(inputs)


# ### Visualising the results

# In[33]:


# Plot the real stock prices and the predicted stock prices
plt.figure(figsize=(14, 7))
sns.set(style="darkgrid")

# Plot the real stock prices with dots and lines
plt.plot(real_stock_price, color='blue', marker='o', label='Real Google Stock Price')

# Plot the predicted stock prices with dots and lines
plt.plot(predicted_stock_price, color='red', marker='o', label='Predicted Google Stock Price')

# Add title and labels to the plot
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')

# Show the legend
plt.legend()

# Display the plot
plt.show()

