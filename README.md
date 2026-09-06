# NFLX Stock Closing Price Predictor

A Django web application that estimates Netflix (NFLX) closing prices from daily Open, High, and Low prices.

## How It Works

The web UI is served by Django. When a user submits the prediction form:

1. `NFLXapp/views.py` validates the three price inputs.
2. `NFLXapp/prediction.py` downloads historical NFLX data from Yahoo Finance.
3. The Open, High, and Low values are min-max normalized using the historical ranges.
4. A pre-trained PyTorch linear regression model predicts a normalized Close value.
5. The prediction is converted back to dollars using the historical Close range.
6. The rounded value is rendered in the results page.

The optional date field is displayed with the result but is not used by the model.

## Deployed Model

The application uses [ml_models/nflx_model.pth](ml_models/nflx_model.pth).
It contains the weights for a single PyTorch linear layer, `nn.Linear(3, 1)`:

- Inputs: Open, High, Low
- Output: normalized closing price
- Model type: linear regression

The model is loaded and used in [NFLXapp/prediction.py](NFLXapp/prediction.py). The repository does not need training notebooks to serve predictions.

## Quick Start

### Requirements

- Python 3.8 or newer
- pip
- Internet access for Yahoo Finance data

### Windows PowerShell

```powershell
git clone https://github.com/AVB934/NFLX.git
cd NFLX
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:SECRET_KEY = "django-insecure-development-key"
python manage.py check
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in a browser.

### macOS/Linux

```bash
git clone https://github.com/AVB934/NFLX.git
cd NFLX
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SECRET_KEY="django-insecure-development-key"
python manage.py check
python manage.py migrate
python manage.py runserver
```

## Using the Application

1. Open the home page and choose the prediction page.
2. Enter positive Open, High, and Low prices.
3. Ensure `High >= Open >= Low`.
4. Submit the form to display the estimated closing price.

The server rejects non-numeric, non-positive, inconsistent, or unusually high values.

## Project Structure

```text
NFLX/
├── NFLXapp/
│   ├── prediction.py       # Model loading, normalization, and prediction
│   ├── views.py            # Form handling and validation
│   ├── templates/          # Home and prediction pages
│   └── static/             # CSS
├── NFLXproject/
│   ├── settings.py         # Django configuration
│   └── urls.py             # Root routes
├── ml_models/
│   ├── nflx_model.pth      # Deployed model weights
├── manage.py               # Django command-line entry point
├── requirements.txt        # Python dependencies
└── GETTING_STARTED.md      # Extended setup guide
```

`db.sqlite3` is created locally by `python manage.py migrate` and is intentionally not committed.

## Development Checks

```bash
python manage.py check
python manage.py test
```

The prediction path requires a network connection because it retrieves current historical ranges from Yahoo Finance. The included model is for demonstration and should not be treated as financial advice.
