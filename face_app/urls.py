from django.urls import path
from . import views

app_name = 'face_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('add_user/', views.add_user, name='add_user'),
    path('capture_image/', views.capture_image, name='capture_image'),
    path('train_model/', views.train_model, name='train_model'),
    path('recognize/', views.recognize_face, name='recognize_face'),
    path('recognize_from_video/', views.recognize_from_video, name='recognize_from_video'),
    path('upload_image/', views.upload_image, name='upload_image'),
    path('recognize_from_image/', views.recognize_from_image, name='recognize_from_image'),
    path('reset_database/', views.reset_database, name='reset_database'),
    path('delete_person/<int:person_id>/', views.delete_person, name='delete_person'),
] 