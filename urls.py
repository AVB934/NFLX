#NFLXproject urls.py
from django.contrib import admin
from django.urls import path, include
from django.views.generic.base import RedirectView

# Root URL configuration
urlpatterns = [
    path('admin/', admin.site.urls),  # Admin panel URL
    path('nflx/', include('NFLXapp.urls')),  # Includes all URLs from the NFLXapp app
    path('', RedirectView.as_view(url='/nflx/')),  # Redirects the root URL (home) to the NFLX app
]