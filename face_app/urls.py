from django.urls import path
from . import views

app_name = 'face_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('add-user/', views.add_user, name='add_user'),
    path('train-model/', views.train_model, name='train_model'),
    path('capture-image/', views.capture_image, name='capture_image'),
    path('recognize-face/', views.recognize_face, name='recognize_face'),
    path('upload-image/', views.upload_image, name='upload_image'),
    path('recognize-from-image/', views.recognize_from_image, name='recognize_from_image'),
] 