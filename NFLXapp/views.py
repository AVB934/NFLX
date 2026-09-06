#NFLXapp views.py
from django.shortcuts import render
from .prediction import predict_stock_price
from datetime import datetime


# View for rendering the home page
def home_view(request):
    return render(request, 'home.html')  # Renders the home.html template

def predict_view(request):
    prediction = None  # Initialize prediction variable
    error_message = None  # Initialize error_message variable

    if request.method == 'POST':  # Check if the request is a POST (form submission)
        try:
            # Extract features from the POST request (form input)
            open_price = float(request.POST['open'])
            high_price = float(request.POST['high'])
            low_price = float(request.POST['low'])
            
            # Validate input values
            if open_price <= 0 or high_price <= 0 or low_price <= 0:
                error_message = "All prices must be positive values."
                return render(request, 'predict.html', {'error_message': error_message})
            
            if high_price < open_price or open_price < low_price:
                error_message = "Invalid price relationship: High must be >= Open, and Open must be >= Low."
                return render(request, 'predict.html', {'error_message': error_message})
            
            if open_price > 10000 or high_price > 10000 or low_price > 10000:
                error_message = "Price values seem unreasonably high. Please check your inputs."
                return render(request, 'predict.html', {'error_message': error_message})
            
            features = [open_price, high_price, low_price]

            # Extract optional date input
            date_str = request.POST.get('date', None)

            # Make a prediction using the extracted features
            prediction = predict_stock_price(features)
            prediction = round(prediction, 2)  # Round the prediction to two decimal places

            # Format the output message based on date input
            if date_str:
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    date_range = f" for {date}"
                except ValueError:
                    date_range = " (invalid date format)"
            else:
                date_range = ""

            return render(request, 'predict.html', {
                'prediction': prediction,
                'date_range': date_range,
            })

        except ValueError as e:
            # Handle the case where the user input is invalid (e.g., non-numeric values)
            error_message = "Invalid input. Please enter valid numbers for all fields."
            return render(request, 'predict.html', {'error_message': error_message})

    return render(request, 'predict.html')
