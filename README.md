# Netflix Stock Price Prediction
Overview

This repository contains two linear regression models for predicting Netflix’s stock prices using historical data. The models are built using PyTorch and are designed to forecast two different target variables:

Adjusted Closing Price: The first model predicts Netflix’s adjusted closing price based on historical stock data.
Regular Closing Price: The second model forecasts the regular closing price.

Both models are implemented with preprocessing, training, and evaluation steps to showcase their performance and accuracy.

Data

The dataset used is NFLX_DATA2023.csv, which includes the following columns:

Open: Opening stock price
High: Highest stock price of the day
Low: Lowest stock price of the day
Volume: Number of shares traded
Close: Regular closing stock price (for the second model)
Adj Close: Adjusted closing stock price (for the first model)

Preprocessing

Data normalisation is performed as follows:

	•	Adjusted Closing Price Model: Min-max normalisation using MinMaxScaler.
	•	Regular Closing Price Model: Min-max normalisation using manual scaling.

The dataset is split into training and test sets with an 80-20 ratio for both models.

Models

1. Adjusted Closing Price Model

	•	Description: Predicts Netflix’s adjusted closing price.
	•	Implementation: Linear regression model in PyTorch.
	•	Training Epochs: 800
	•	Metrics:
	•	Test Loss (MSE): 0.0006
	•	Test MAE: 0.0170
	•	Test R-squared: 0.9896
	•	Visualisation:
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
	•	Visualisation:
	•	Training loss over epochs.
	•	Actual vs. predicted values.

Training

Both models are trained using stochastic gradient descent (SGD) with a learning rate of 0.01. The training process involves monitoring loss reduction over epochs to ensure model learning and performance improvement.

Evaluation

The performance of each model is evaluated using the following metrics:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R-squared Value

These metrics are reported to assess model accuracy and predictive capability.

Results

Adjusted Closing Price Model: Achieves high accuracy with a low MSE and MAE, and a high R-squared value.
Regular Closing Price Model: Shows even better performance with slightly lower MSE and MAE, and a higher R-squared value.

Conclusion

The linear regression models effectively predict Netflix’s stock prices, with each model tailored to different target variables. The high performance metrics demonstrate the models’ capability in making accurate predictions. Further validation and potential enhancements are recommended for practical applications.
