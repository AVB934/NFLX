# Getting Started - NFLX Stock Price Predictor

Complete step-by-step guide to run the Netflix Stock Price Prediction application locally.

---

## 🐍 Your Virtual Environment Details

Your project uses a **Python virtual environment** to isolate dependencies. Here are the details:

**Location:** `d:\Documents\Projects\NFLX\venv`

**Python Version:** 3.13.9

**Environment Type:** Standard venv (Python Virtual Environment)

**How to Activate:**
```bash
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# macOS/Linux:
source venv/bin/activate
```

**How to Deactivate:**
```bash
deactivate
```

**What it Contains:**
- Django 6.0.3 (web framework)
- PyTorch 2.10.0 (ML framework)
- scikit-learn (data preprocessing)
- pandas (data manipulation)
- yfinance (stock data API)
- numpy (numerical computing)

**Purpose:**
- Isolates project dependencies from system Python
- Prevents conflicts with other Python projects
- Makes the project portable and reproducible
- Easy to share and deploy

---

Before you start, make sure you have:
- **Python 3.8+** installed ([Download Python](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Git** (optional, for cloning the repo)
- **Internet connection** (for downloading dependencies and Yahoo Finance data)

### Verify Installation
```bash
python --version
pip --version
```

---

## ⚡ Quick Terminal Commands (Copy & Paste)

### First Time Setup
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:SECRET_KEY = "django-insecure-test-key"
python manage.py check
$env:SECRET_KEY = "django-insecure-test-key"; python manage.py runserver
```

### Every Time After (Shorter)
```bash
.\venv\Scripts\Activate.ps1
$env:SECRET_KEY = "django-insecure-test-key"
python manage.py runserver
```

### Then Open Browser
```
http://localhost:8000/
```

---

## Step 1: Get the Project

### Option A: Clone from GitHub
```bash
git clone https://github.com/AVB934/NFLX.git
cd NFLX
```

### Option B: Download as ZIP
1. Go to [GitHub Repository](https://github.com/AVB934/NFLX)
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file
4. Open PowerShell/Terminal in the extracted folder

---

## Step 2: Create Virtual Environment

Creating a virtual environment isolates project dependencies and prevents conflicts.

### Windows (PowerShell)
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**If you get an execution policy error:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try Activate.ps1 again
```

### macOS/Linux (Bash/Zsh)
```bash
python3 -m venv venv
source venv/bin/activate
```

**Expected output:** You should see `(venv)` prefix in your terminal:
```
(venv) ~/NFLX>
```

---

## Step 3: Install Dependencies

With virtual environment activated, install all required packages:

### Windows (PowerShell)
```bash
pip install -r requirements.txt
```

### macOS/Linux
```bash
pip install -r requirements.txt
```

**This installs:**
- Django (web framework)
- PyTorch (ML framework)
- scikit-learn (data preprocessing)
- pandas (data manipulation)
- yfinance (stock data)
- numpy (numerical computing)

**Expected:** Installation should complete without errors. You should see:
```
Successfully installed Django-6.0.3 torch-2.10.0 ...
```

---

## Step 4: Set Environment Variables

The application needs a SECRET_KEY for security. Set it before running:

### Windows (PowerShell)
```bash
$env:SECRET_KEY = "django-insecure-test-key"
```

### macOS/Linux (Bash/Zsh)
```bash
export SECRET_KEY="django-insecure-test-key"
```

> **Note:** For production, use a strong random key. See [Django SECRET_KEY Docs](https://docs.djangoproject.com/en/6.0/ref/settings/#secret-key)

---

## Step 5: Verify Django Setup

Before running the server, check that everything is configured correctly:

```bash
python manage.py check
```

**Expected output:**
```
System check identified no issues (0 silenced).
```

If you see errors, fix them before proceeding to Step 6.

---

## 🚀 Step 6: Run the Development Server

Start the Django development server:

### Windows (PowerShell)
```bash
$env:SECRET_KEY = "django-insecure-test-key"
python manage.py runserver
```

### macOS/Linux (Bash/Zsh)
```bash
export SECRET_KEY="django-insecure-test-key"
python manage.py runserver
```

**Expected output:**
```
Watching for file changes with StatReloader
System check identified no issues (0 silenced).
March 15, 2026 - 10:00:07
Django version 6.0.3, using settings 'NFLXproject.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.

WARNING: This is a development server. Do not use it in a production setting.
```

✅ **Server is now running!**

---

## 🌐 Step 7: Access the Application

Open your web browser and navigate to:

**http://localhost:8000/**

You should see:
- ✅ NFLX Stock Closing Price Prediction home page
- ✅ "Closing Price Predictor" header with navigation
- ✅ "Get Started" button
- ✅ Clean, professional styling

---

## 🧪 Step 8: Test the Application

### Basic Prediction Test

1. Click **"Get Started"** or click **"Predict"** in the navigation

2. Enter test stock prices:
   ```
   Open:  150.00
   High:  155.00
   Low:   145.00
   Date:  (leave empty or select today)
   ```

3. Click **"Predict"**

4. You should see a result like:
   ```
   Predicted Closing Price
   $147.71
   ```

### Input Validation Tests

Try these invalid inputs to verify error handling:

**Test 1: Invalid Price Relationship**
```
Open:  150
High:  145     ❌ (High must be ≥ Open)
Low:   160
```
Expected: Error message about price relationship

**Test 2: Negative Values**
```
Open:  -100    ❌ (Must be positive)
High:  50
Low:   10
```
Expected: Error message about positive values

**Test 3: Unreasonably High Values**
```
Open:  50000   ❌ (Exceeds limit)
High:  55000
Low:   45000
```
Expected: Error message about high values

---

## 🛑 Step 9: Stop the Server

When you're done testing:

**Press:** `CTRL + BREAK` (Windows) or `CTRL + C` (macOS/Linux)

```
KeyboardInterrupt
Quit the server with CTRL-BREAK.
(venv) PS D:\Documents\Projects\NFLX>
```

---

## 🔄 Next Time You Run the App

You only need to do Steps 4-6:

```bash
# Navigate to project folder
cd path\to\NFLX

# Activate venv
.\venv\Scripts\Activate.ps1

# Set environment variable
$env:SECRET_KEY = "django-insecure-test-key"

# Run server
python manage.py runserver
```

---

## 🐛 Troubleshooting

### ❌ Error: "ModuleNotFoundError: No module named 'django'"

**Cause:** Virtual environment not activated or dependencies not installed

**Fix:**
```bash
.\venv\Scripts\Activate.ps1      # Activate venv
pip install -r requirements.txt   # Install dependencies
```

---

### ❌ Error: "The SECRET_KEY setting must not be empty"

**Cause:** Environment variable not set

**Fix:**
```bash
$env:SECRET_KEY = "django-insecure-test-key"
python manage.py runserver
```

---

### ❌ Error: "Port 8000 is already in use"

**Cause:** Another application is using port 8000

**Fix:** Use a different port:
```bash
python manage.py runserver 8001
# Then visit: http://localhost:8001/
```

---

### ❌ Error: "This site can't be reached - ERR_CONNECTION_REFUSED"

**Cause:** Server not running or port is different

**Fix:**
1. Check the terminal where you ran `python manage.py runserver`
2. Look for line: `Starting development server at http://127.0.0.1:XXXX/`
3. Visit that exact address in your browser
4. Restart the server if needed

---

### ❌ Error: "No module named 'torch'"

**Cause:** Incomplete dependency installation

**Fix:**
```bash
pip install torch scikit-learn pandas yfinance --upgrade
```

---

### ❌ Website looks broken (no colors, misaligned text)

**Cause:** Static files not loading (rare in development)

**Fix:**
```bash
python manage.py collectstatic --noinput
python manage.py runserver
```

---

## 📝 Common Commands

| Command | Purpose |
|---------|---------|
| `python manage.py runserver` | Start development server |
| `python manage.py check` | Verify Django setup |
| `python manage.py shell` | Django interactive shell |
| `deactivate` | Exit virtual environment |
| `pip list` | Show installed packages |
| `pip install -r requirements.txt` | Install all dependencies |

---

## 📚 Additional Resources

- [Django Documentation](https://docs.djangoproject.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Project Structure](PROJECT_STRUCTURE.md)
- [README](README.md)
- [ML Models Info](ml_models/README.md)

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (no errors)
- [ ] SECRET_KEY environment variable set
- [ ] `python manage.py check` shows no issues
- [ ] Server running on http://localhost:8000/
- [ ] Home page loads in browser
- [ ] Prediction form works with test data
- [ ] Input validation working (error messages appear)

---

## 🎉 You're All Set!

Your NFLX Stock Price Predictor is now running locally. 

**Next Steps:**
- Experiment with different stock prices
- Review the [Project Structure](PROJECT_STRUCTURE.md)
- Check out the ML models in `ml_models/`
- Read the [README](README.md) for more information

**Questions?** Check the troubleshooting section above or review the main README.

Happy predicting! 🚀
