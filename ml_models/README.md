# Deployed Model

This directory contains the model used by the Django application:

- `nflx_model.pth`: PyTorch state dictionary for a linear regression model.

The model architecture is defined in `NFLXapp/prediction.py` as `nn.Linear(3, 1)`.
It accepts three normalized features:

1. Open price
2. High price
3. Low price

It returns one normalized closing-price prediction. The application then denormalizes that value using the historical NFLX Close range downloaded from Yahoo Finance.

Training notebooks are not included because they are not required to run the web application. To replace the model, create compatible weights with the same three-input, one-output architecture and update `nflx_model.pth`.
