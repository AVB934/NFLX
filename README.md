# Netflix Stock Price Prediction

A full-stack Django web application for predicting Netflix stock closing prices using machine learning models trained on historical data from Yahoo Finance.

## 🎯 Features

- **Web Interface**: User-friendly Django web application to input stock data and get predictions
- **ML Model**: PyTorch linear regression model trained on historical NFLX data
- **Real-Time Data**: Integrates with Yahoo Finance API for live data normalization
- **Input Validation**: Validates stock prices (High ≥ Open ≥ Low, all positive values)
- **Responsive Design**: Clean, modern UI with error handling

## 📋 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AVB934/NFLX.git
   cd NFLX
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   # Windows PowerShell:
   $env:SECRET_KEY = "django-insecure-your-secret-key"
   
   # Or create a .env file and source it
   ```

5. **Run the development server**
   ```bash
   # Windows PowerShell:
   $env:SECRET_KEY = "django-insecure-test-key"
   .\venv\Scripts\python.exe manage.py runserver
   
   # On macOS/Linux:
   export SECRET_KEY="django-insecure-test-key"
   python manage.py runserver
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:8000/`
   
   You should see the NFLX Stock Closing Price Prediction home page with:
   - Navigation header
   - "Get Started" button
   - Clean, professional design

> **For detailed step-by-step instructions, see [GETTING_STARTED.md](GETTING_STARTED.md)**

## 🏗️ Project Structure

```
NFLX/
├── NFLXproject/                # Django project configuration
│   ├── settings.py            # Project settings & configuration
│   ├── urls.py               # Root URL routing
│   ├── asgi.py               # ASGI configuration
│   └── wsgi.py               # WSGI configuration
│
├── NFLXapp/                   # Main Django application
│   ├── templates/            # HTML templates
│   │   ├── base.html        # Base template (header, nav, footer)
│   │   ├── home.html        # Home page
│   │   └── predict.html     # Prediction form & results
│   ├── static/              # Static files
│   │   └── styles.css       # Website styling
│   ├── views.py             # View functions (request handlers)
│   ├── urls.py              # App URL routing
│   ├── prediction.py        # ML model loading & prediction logic
│   ├── models.py            # Database models (currently unused)
│   ├── tests.py             # Unit tests
│   └── apps.py              # App configuration
│
├── ml_models/               # Machine learning models & notebooks
│   ├── nflx_model.pth      # Pre-trained PyTorch linear regression model
│   ├── NFLX.ipynb          # General analysis notebook
│   ├── NFLX_ClosingPrice_Pred.ipynb      # Closing price model
│   ├── NFLX_ADJClosingPrice_Pred.ipynb   # Adjusted closing price model
│   ├── NFLX_LSTM.ipynb     # LSTM model notebook
│   └── README.md           # ML models documentation
│
├── manage.py               # Django management utility
├── requirements.txt        # Python dependencies
├── db.sqlite3              # SQLite database
├── README.md               # This file
├── PROJECT_STRUCTURE.md    # Detailed project structure
└── .gitignore              # Git ignore file
```

## 🚀 Usage

### Via Web Interface

1. Navigate to `http://localhost:8000/`
2. Click "Get Started" on the home page
3. Enter stock prices:
   - **Open**: Opening price of the day (USD)
   - **High**: Highest price of the day (USD)
   - **Low**: Lowest price of the day (USD)
   - **Date** (optional): Date for reference
4. Click "Predict" to get the predicted closing price
5. Result displays the predicted closing price

### Input Validation

The app validates all inputs:
- ✓ All prices must be positive numbers
- ✓ High price ≥ Open price
- ✓ Open price ≥ Low price
- ✓ No unreasonably high values (> $10,000)

## 🧠 Machine Learning Model

### Current Model (Deployed)
- **Type**: Linear Regression (PyTorch)
- **Input Features**: 3 (Open, High, Low prices)
- **Output**: Predicted closing price (normalized)
- **Training Data**: Historical NFLX data from Yahoo Finance (2002-present)
- **Normalization**: Min-Max scaling

### Model Performance
| Metric | Value |
|--------|-------|
| Test MSE | 0.0006 |
| Test MAE | 0.0170 |
| R-squared | 0.9896 |

### Alternative Models Available
- LSTM model (superior performance, R² = 0.9987)
- Regular closing price prediction model
- Adjusted closing price prediction model

See `ml_models/README.md` for detailed model information and training notebooks.

## 🔧 Configuration

### Django Settings
Edit `NFLXproject/settings.py` to modify:
- `DEBUG`: Set to `False` for production
- `ALLOWED_HOSTS`: Add your deployment domain
- `SECRET_KEY`: Use environment variable in production
- `DATABASES`: Configure database (currently SQLite)

### Environment Variables
```bash
SECRET_KEY = "your-secret-key"  # Required for production
```

## 📊 Data Flow

1. User enters stock prices (Open, High, Low)
2. App validates input values
3. Fetches historical NFLX data from Yahoo Finance
4. Normalizes input using min-max scaling
5. Passes normalized features to PyTorch model
6. Model outputs normalized prediction
7. Denormalizes prediction using historical Close price data
8. Displays predicted closing price to user

## 🔐 Security Features

- ✓ CSRF protection on form submissions
- ✓ Input validation on both client and server
- ✓ SECRET_KEY stored in environment variable
- ✓ SQLi and XSS protections via Django ORM
- ✓ Debug mode disabled for production

## 🐛 Troubleshooting

### Issue: Model file not found
**Solution**: Ensure `nflx_model.pth` is in the `ml_models/` folder

### Issue: Yahoo Finance connection errors
**Solution**: Check internet connection; the app will retry automatically

### Issue: SECRET_KEY error on startup
**Solution**: Set the `SECRET_KEY` environment variable before running

### Issue: Static files not loading
**Solution**: Run `python manage.py collectstatic` in production

## 📈 Future Improvements

- [ ] Store prediction history in database
- [ ] Add REST API endpoints
- [ ] Implement model version management
- [ ] Add more advanced models (LSTM, GRU)
- [ ] Create admin dashboard
- [ ] Add model retraining pipeline
- [ ] Implement caching for performance

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created by AVB934

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.
