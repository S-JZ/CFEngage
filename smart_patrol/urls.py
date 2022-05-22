from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('start-cam', views.start_cam, name='cam')
]