# NFLX
This project implements a linear regression model to predict Netflix’s adjusted closing stock price using historical stock market data. The primary objective is to forecast future stock prices based on features such as the opening price, highest price, lowest price, and trading volume.

Data

The dataset used for this project is sourced from NFLX_DATA2023.csv, which contains historical stock market data for Netflix. The features included are:

Open: Opening stock price
High: Highest stock price of the day
Low: Lowest stock price of the day
Volume: Number of shares traded
Adj Close: Adjusted closing stock price (target variable)

Preprocessing

Data normalisation is performed using MinMaxScaler to scale features between 0 and 1, which helps in improving the performance of the linear regression model. The dataset is split into training and test sets with an 80-20 ratio.

Model

A simple linear regression model is built using PyTorch. The model consists of a single linear layer that predicts the adjusted closing stock price based on the input features.

Training

The model is trained using stochastic gradient descent (SGD) with a learning rate of 0.01. The training process involves 800 epochs, during which the model’s loss is monitored and minimised.

Evaluation

The model’s performance is evaluated on the test set using the following metrics:

Mean Squared Error (MSE): 0.0006
Mean Absolute Error (MAE): 0.0170
R-squared: 0.9896

These metrics indicate that the model performs exceptionally well, with predictions closely matching the actual values.

Results

The training loss decreased significantly over epochs, demonstrating effective learning. The test metrics reflect high accuracy, with a low mean squared error, mean absolute error, and a high R-squared value, indicating that the model explains almost 99% of the variance in the test data.
