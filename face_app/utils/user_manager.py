"""
User management and face capture functionality.
"""
import cv2
import os
import time
import json
from django.conf import settings
from .definitions import FRAME_WIDTH, FRAME_HEIGHT, NUM_IMAGES, WHITE

class DatabaseManager:
    def __init__(self):
        self.is_capturing = False

    def insert_or_update(self, name):
        """Insert or update a user in the database and return the user ID."""
        from face_app.models import Person
        person, created = Person.objects.get_or_create(name=name)
        return person.id

    def create_dataset_folder(self, id, name):
        """Create a folder for the user's face dataset."""
        dataset_folder = os.path.join(settings.MEDIA_ROOT, 'dataSet')
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        user_folder = os.path.join(dataset_folder, f"{name}_{id}")
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)

        return user_folder

    def capture_faces(self, user_folder, id, frame):
        """Capture faces from a single frame and save if face detected.
        
        In Django context, we'll process single frames sent from the client.
        """
        detector = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, "face_app/model/haarcascade_frontalface_default.xml"))
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector.detectMultiScale(gray, 1.3, 5)
        
        result = {
            "faces_detected": len(faces),
            "image_saved": False,
            "file_path": None
        }
        
        if len(faces) > 0:
            # We only process the first face detected
            x, y, w, h = faces[0]
            
            # Get existing images count for this user
            existing_files = [f for f in os.listdir(user_folder) if f.endswith('.jpg')]
            sample_num = len(existing_files) + 1
            
            # Save the face image
            file_path = os.path.join(user_folder, f"User.{id}.{sample_num}.jpg")
            cv2.imwrite(file_path, gray[y:y+h, x:x+w])
            
            result["image_saved"] = True
            result["file_path"] = os.path.relpath(file_path, settings.MEDIA_ROOT)
        
        return result

    def count_user_images(self, user_id, name):
        """Count the number of images for a specific user."""
        user_folder = os.path.join(settings.MEDIA_ROOT, 'dataSet', f"{name}_{user_id}")
        if os.path.exists(user_folder):
            image_files = [f for f in os.listdir(user_folder) if f.endswith('.jpg')]
            return len(image_files)
        return 0 