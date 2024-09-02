Netflix Stock Price Prediction

This repository contains three distinct models for predicting Netflix’s stock prices using historical data. The models employ different methodologies and are implemented in TensorFlow/Keras and PyTorch.

Data

The dataset used is NFLX_DATA2023.csv, which includes the following columns:

•	Open: Opening stock price
•	High: Highest stock price of the day
•	Low: Lowest stock price of the day
•	Volume: Number of shares traded
•	Close: Regular closing stock price (for the second model)
•	Adj Close: Adjusted closing stock price (for the first and third models)

Preprocessing

Data normalization is performed as follows:

•	Adjusted Closing Price Model: Min-max normalization using MinMaxScaler.
•	Regular Closing Price Model: Min-max normalization using manual scaling.

The dataset is split into training and test sets with an 80-20 ratio for all models.

Models

1. Adjusted Closing Price Model

•	Description: Predicts Netflix’s adjusted closing price.
•	Implementation: Linear regression model in PyTorch.
•	Training Epochs: 800
•	Metrics:
•	Test Loss (MSE): 0.0006
•	Test MAE: 0.0170
•	Test R-squared: 0.9896
•	Visualizations:
•	Training loss over epochs
•	Actual vs. predicted values

2. Regular Closing Price Model

•	Description: Forecasts Netflix’s regular closing price.
•	Implementation: Linear regression model in PyTorch.
•	Training Epochs: 1000
•	Metrics:
•	Test Loss (MSE): 0.0005
•	Test MAE: 0.0149
•	Test R-squared: 0.9915
•	Visualizations:
•	Training loss over epochs
•	Actual vs. predicted values

3. Adjusted Closing Price Model with LSTM

•	Description: Predicts Netflix’s adjusted closing price using Long Short-Term Memory (LSTM) neural network.
•	Implementation: LSTM model in TensorFlow/Keras.
•	Training Epochs: 50
•	Metrics:
•	Test Loss (MSE): 0.0001
•	Test MAE: 0.0042
•	Test R-squared: 0.9987
•	Visualizations:
•	Training and validation loss over epochs
•	Actual vs. predicted values

Training

•	Linear Regression Models: Trained using Stochastic Gradient Descent (SGD) with a learning rate of 0.01.
•	LSTM Model: Trained using the Adam optimizer.

Training involves monitoring the reduction of loss over epochs to ensure model learning and performance improvement.

Evaluation

Model performance is assessed using the following metrics:

•	Mean Squared Error (MSE)
•	Mean Absolute Error (MAE)
•	R-squared Value

These metrics evaluate the accuracy and predictive capability of the models.

Results

•	Adjusted Closing Price Model: High accuracy with low MSE and MAE, and a high R-squared value.
•	Regular Closing Price Model: Excellent performance with slightly lower MSE and MAE, and a higher R-squared value.
•	LSTM Model: Superior performance with extremely low MSE and MAE, and a very high R-squared value.

Conclusion

The models effectively predict Netflix’s stock prices, with each model tailored to different target variables. The high performance metrics of the LSTM model suggest its effectiveness in capturing temporal patterns for stock price prediction.

Full-Stack Machine Learning Web Application

In addition to the models, a full-stack web application has been developed using historical data sourced from Yahoo Finance:

•	Back-End: Built with Python and PyTorch, utilizing Linear Regression and LSTM models for time series forecasting.
•	Front-End: Developed using Django, featuring an interactive interface for users to input stock data (Open, High, Low) and obtain predictions of future closing prices.
•	Real-Time Data Updates: Integrated with Yahoo Finance for real-time data.
•	Visualization: Utilizes Matplotlib to visualize predicted vs. actual stock prices.
•	Model Evaluation: Performance is evaluated through metrics such as Mean Squared Error (MSE) and R-squared scores.
