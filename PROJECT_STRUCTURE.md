# Project Structure Overview

```
NFLX/
├── NFLXproject/                    # Django project configuration
│   ├── __init__.py
│   ├── settings.py                 # Project settings, templates, and static files config
│   ├── urls.py                     # Root URL routing
│   ├── asgi.py                     # ASGI configuration
│   └── wsgi.py                     # WSGI configuration
│
├── NFLXapp/                        # Django app - main application
│   ├── migrations/                 # Database migrations
│   │   └── __init__.py
│   ├── templates/                  # HTML templates
│   │   ├── base.html               # Base template (header, nav, footer)
│   │   ├── home.html               # Home page
│   │   └── predict.html            # Prediction page
│   ├── static/                     # Static files
│   │   └── styles.css              # Website styling
│   ├── __init__.py
│   ├── admin.py                    # Django admin configuration
│   ├── apps.py                     # App configuration
│   ├── models.py                   # Database models
│   ├── tests.py                    # Unit tests
│   ├── urls.py                     # App URL routing
│   ├── views.py                    # View functions
│   └── prediction.py               # ML model prediction logic
│
├── ml_models/                      # Deployed machine learning model
│   ├── README.md                   # Model documentation
│   └── nflx_model.pth              # Pre-trained PyTorch linear model
│
├── manage.py                       # Django management utility
├── db.sqlite3                      # Local database created by migrate (ignored)
├── requirements.txt                # Python dependencies
├── README.md                       # Project README
├── .gitignore                      # Git ignore file
└── .git/                           # Git repository

```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
# Windows PowerShell
$env:SECRET_KEY = "your-secure-key-here"

# Or create a .env file and use python-dotenv
```

### 3. Run the Development Server
```bash
python manage.py runserver
```

Access the app at `http://localhost:8000/`

## Key Files

| File | Purpose |
|------|---------|
| `NFLXproject/settings.py` | Django configuration, template/static paths |
| `NFLXapp/views.py` | Request handlers and business logic |
| `NFLXapp/prediction.py` | ML model loading and prediction logic |
| `NFLXapp/templates/predict.html` | Prediction form UI |
| `NFLXapp/static/styles.css` | Website styling |
| `ml_models/nflx_model.pth` | Pre-trained model |

## File Organization Benefits

✓ **Separation of Concerns**: Config, logic, and resources in separate folders
✓ **Django Best Practices**: Standard app structure
✓ **Scalability**: Easy to add more apps, models, or templates
✓ **Maintainability**: Clear organization for future development
✓ **Version Control**: .gitignore prevents tracking unnecessary files

