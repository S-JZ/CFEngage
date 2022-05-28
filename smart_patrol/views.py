from django.shortcuts import redirect, render
from .forms import UploadFileForm
from django.views.decorators.csrf import ensure_csrf_cookie
from .FaceRecognition import CamPolice



@ensure_csrf_cookie
def home(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            return redirect('cam')
    else:
        form = UploadFileForm()
    return render(request, "index.html", {'form': form})


def start_cam(request):
    file = request.FILES['file']
    path = file.temporary_file_path()
    CamPolice(path).process_video()
    return render(request, "index.html")


def about(request):
    return render(request, "about.html")
