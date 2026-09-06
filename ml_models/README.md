# ML Models & Jupyter Notebooks

This folder contains:

1. **nflx_model.pth** - Pre-trained PyTorch linear regression model for NFLX stock price prediction
2. **Jupyter Notebooks** - Training notebooks for the various models:
   - NFLX.ipynb
   - NFLX_ClosingPrice_Pred.ipynb
   - NFLX_ADJClosingPrice_Pred.ipynb
   - NFLX_LSTM.ipynb

## Model Information

The currently deployed model (`nflx_model.pth`) is a linear regression model trained on historical NFLX data with 3 input features:
- Open price
- High price
- Low price

**Output:** Predicted closing price (normalized between 0-1)

## To Retrain the Model

Run any of the Jupyter notebooks to retrain and save a new model. The notebook will:
1. Download historical data
2. Preprocess and normalize the data
3. Train the model
4. Save the trained model as `nflx_model.pth`
