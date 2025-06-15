"""
Views for the face recognition app.
"""
import cv2
import numpy as np
import os
import json
import base64
import shutil
from PIL import Image
from io import BytesIO

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.urls import reverse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .models import Person, CapturedImage, RecognizedFace
from .utils.user_manager import DatabaseManager
from .utils.train_model import TrainModel
from .utils.face_recognizer import FaceRecognizer
from .utils.image_recognition import FaceRecognitionFromImage
from .utils.definitions import FRAME_WIDTH, FRAME_HEIGHT

# Initialize components
db_manager = DatabaseManager()
face_trainer = TrainModel()
face_recognizer = FaceRecognizer()

def index(request):
    """Home page view."""
    people = Person.objects.all()
    context = {
        'people': people,
    }
    return render(request, 'face_app/index.html', context)

def add_user(request):
    """View for adding a new user to the system."""
    if request.method == 'POST':
        name = request.POST.get('name')
        if name:
            new_id = db_manager.insert_or_update(name)
            user_folder = db_manager.create_dataset_folder(new_id, name)
            
            return render(request, 'face_app/capture_images.html', {
                'person_id': new_id,
                'person_name': name,
                'image_count': 0
            })
    
    return render(request, 'face_app/add_user.html')

def reset_database(request):
    """Reset database and remove all user data."""
    if request.method == 'POST':
        # Delete all person records
        Person.objects.all().delete()
        
        # Remove dataset directory
        dataset_path = os.path.join(settings.MEDIA_ROOT, 'dataSet')
        if os.path.exists(dataset_path):
            try:
                shutil.rmtree(dataset_path)
                os.makedirs(dataset_path)  # Create empty directory again
            except Exception as e:
                print(f"Error deleting dataset directory: {e}")
        
        # Remove temp directory
        temp_path = os.path.join(settings.MEDIA_ROOT, 'temp')
        if os.path.exists(temp_path):
            try:
                shutil.rmtree(temp_path)
                os.makedirs(temp_path)  # Create empty directory again
            except Exception as e:
                print(f"Error deleting temp directory: {e}")
        
        # Delete training model file
        model_path = os.path.join(settings.BASE_DIR, 'model', 'trainner.yml')
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except Exception as e:
                print(f"Error deleting model file: {e}")
                
        return redirect('face_app:index')
    
    return render(request, 'face_app/reset_confirm.html')

def delete_person(request, person_id):
    """Delete a person and their data."""
    try:
        person = get_object_or_404(Person, id=person_id)
        person_name = person.name
        
        # Delete person's dataset folder
        user_folder = os.path.join(settings.MEDIA_ROOT, 'dataSet', f"{person_name}_{person_id}")
        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)
            
        # Delete from database
        person.delete()
        
        # Return success response if AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True})
            
        return redirect('face_app:index')
    except Exception as e:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
        return redirect('face_app:index')

@csrf_exempt
def capture_image(request):
    """API endpoint for capturing face images."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            person_id = int(data.get('id'))
            person_name = data.get('name')
            image_data = data.get('image')
            direction = data.get('direction', 'thang')  # Default to 'thang' if not specified
            
            # Get person object
            person = get_object_or_404(Person, id=person_id)
            
            # Create user folder if not exists
            user_folder = db_manager.create_dataset_folder(person_id, person_name)
            
            # Convert base64 to image
            format, imgstr = image_data.split(';base64,') 
            ext = format.split('/')[-1]
            image_data = base64.b64decode(imgstr)
            
            # Convert to CV2 format
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Flip image like in the original code
            frame = cv2.flip(frame, 1)
            
            # Draw rectangle in center for face positioning guide
            centerH = frame.shape[0] // 2
            centerW = frame.shape[1] // 2
            cv2.rectangle(
                frame,
                (centerW - FRAME_WIDTH // 2, centerH - FRAME_HEIGHT // 2),
                (centerW + FRAME_WIDTH // 2, centerH + FRAME_HEIGHT // 2),
                (255, 255, 255),
                5
            )
            
            # Show direction guidance
            cv2.putText(
                frame,
                direction,
                (centerW - 100, centerH - FRAME_HEIGHT // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            
            # Convert to gray for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process and save the image
            result = db_manager.capture_faces(user_folder, person_id, frame)
            
            # Get updated count
            image_count = db_manager.count_user_images(person_id, person_name)
            
            return JsonResponse({
                'success': True,
                'faces_detected': result['faces_detected'],
                'image_saved': result['image_saved'],
                'image_count': image_count,
                'file_path': result['file_path'] if result['image_saved'] else None
            })
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=400)

def train_model(request):
    """View for training the face recognition model."""
    success, message = face_trainer.train()
    
    context = {
        'success': success,
        'message': message
    }
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse(context)
    
    return render(request, 'face_app/train_model.html', context)

def recognize_face(request):
    """View for the face recognition interface."""
    return render(request, 'face_app/recognize.html')

@csrf_exempt
def recognize_from_video(request):
    """API endpoint for recognizing faces from video frames."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            
            # Convert base64 to image
            format, imgstr = image_data.split(';base64,') 
            ext = format.split('/')[-1]
            image_data = base64.b64decode(imgstr)
            
            # Convert to CV2 format
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the frame
            processed_frame, results = face_recognizer.recognize_face_in_frame(frame)
            
            # Convert back to base64 for response
            _, buffer = cv2.imencode(f'.{ext}', processed_frame)
            processed_image = f"data:image/{ext};base64," + base64.b64encode(buffer).decode('utf-8')
            
            return JsonResponse({
                'success': True,
                'processed_image': processed_image,
                'results': results
            })
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=400)

def upload_image(request):
    """View for uploading an image for face recognition."""
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Save the uploaded image
        path = default_storage.save(f'temp/{image_file.name}', ContentFile(image_file.read()))
        full_path = os.path.join(settings.MEDIA_ROOT, path)
        
        # Store the path in the session for recognition
        request.session['uploaded_image_path'] = full_path
        
        context = {
            'uploaded_image': os.path.join(settings.MEDIA_URL, path),
            'ready_for_recognition': True
        }
        
        return render(request, 'face_app/upload_image.html', context)
    
    return render(request, 'face_app/upload_image.html')

def recognize_from_image(request):
    """API endpoint for recognizing faces from an uploaded image."""
    image_path = request.session.get('uploaded_image_path')
    
    if not image_path or not os.path.exists(image_path):
        context = {
            'error': 'No image uploaded or image not found.'
        }
        return render(request, 'face_app/upload_image.html', context)
    
    # Initialize the image recognition
    image_recognizer = FaceRecognitionFromImage()
    processed_image_path, results = image_recognizer.process_image(image_path)
    
    context = {
        'original_image': os.path.relpath(image_path, settings.MEDIA_ROOT).replace('\\', '/'),
        'processed_image': processed_image_path.replace('\\', '/') if processed_image_path else None,
        'results': results,
        'ready_for_recognition': False
    }
    
    return render(request, 'face_app/recognition_results.html', context)
