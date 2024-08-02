# Netflix Stock Price Prediction
This repository contains three models for predicting Netflix’s stock prices using historical data. Each model is designed to forecast different target variables and employs distinct methodologies. The models are implemented in TensorFlow/Keras and PyTorch.

Data

The dataset used is NFLX_DATA2023.csv, which includes the following columns:

Open: Opening stock price

High: Highest stock price of the day

Low: Lowest stock price of the day

Volume: Number of shares traded

Close: Regular closing stock price (for the second model)

Adj Close: Adjusted closing stock price (for the first and third models)

Preprocessing

Data normalisation is performed as follows:

Adjusted Closing Price Model: Min-max normalisation using MinMaxScaler.
Regular Closing Price Model: Min-max normalisation using manual scaling.

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
	•	Visualisations:
	•	Training loss over epochs.
	•	Actual vs. predicted values.

2. Regular Closing Price Model

	•	Description: Forecasts Netflix’s regular closing price.
	•	Implementation: Linear regression model in PyTorch.
	•	Training Epochs: 1000
	•	Metrics:
	•	Test Loss (MSE): 0.0005
	•	Test MAE: 0.0149
	•	Test R-squared: 0.9915
	•	Visualisations:
	•	Training loss over epochs.
	•	Actual vs. predicted values.

3. Adjusted Closing Price Model with LSTM

	•	Description: Predicts Netflix’s adjusted closing price using Long Short-Term Memory (LSTM) neural network.
	•	Implementation: LSTM model in TensorFlow/Keras.
	•	Training Epochs: 50
	•	Metrics:
	•	Test Loss (MSE): 0.0001
	•	Test MAE: 0.0042
	•	Test R-squared: 0.9987
	•	Visualisations:
	•	Training and validation loss over epochs.
	•	Actual vs. predicted values.

Training

The models are trained using the following optimisation techniques:

Linear Regression Models: Stochastic Gradient Descent (SGD) with a learning rate of 0.01.
LSTM Model: Adam optimiser.

The training process involves monitoring loss reduction over epochs to ensure model learning and performance improvement.

Evaluation

The performance of each model is evaluated using the following metrics:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R-squared Value

These metrics assess model accuracy and predictive capability.

Results

Adjusted Closing Price Model: Achieves high accuracy with low MSE and MAE, and a high R-squared value.
Regular Closing Price Model: Shows excellent performance with slightly lower MSE and MAE, and a higher R-squared value.
LSTM Model: Demonstrates superior performance with extremely low MSE and MAE, and a very high R-squared value.

Conclusion

The models effectively predict Netflix’s stock prices, with each model tailored to different target variables. The high performance metrics of the LSTM model suggest its effectiveness in capturing temporal patterns for stock price prediction.

