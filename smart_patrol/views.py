from django.shortcuts import render
from .FaceRecognition import CamPolice

def home(request):
    return render(request, "index.html")

def start_cam(request):
    CamPolice.detect_faces()
    return render(request, "index.html")





