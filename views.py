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
            features = [
                float(request.POST['open']),  # Convert 'open' input to float
                float(request.POST['high']),  # Convert 'high' input to float
                float(request.POST['low']),   # Convert 'low' input to float
            ]

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