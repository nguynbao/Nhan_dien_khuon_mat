"""
Model training functionality for face recognition.
"""
import cv2
import os
import numpy as np
from PIL import Image
from django.conf import settings

class TrainModel:
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, "face_app/model/haarcascade_frontalface_default.xml"))

    def get_images_and_labels(self, path):
        """Extract face samples and IDs from dataset folder."""
        face_samples = []
        ids = []

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".jpg"):
                    image_path = os.path.join(root, file)
                    pil_image = Image.open(image_path).convert('L')
                    image_np = np.array(pil_image, 'uint8')
                    folder_name = os.path.basename(root)
                    
                    try:
                        id = int(folder_name.split("_")[1])
                    except ValueError:
                        continue

                    faces = self.detector.detectMultiScale(image_np)
                    for (x, y, w, h) in faces:
                        face_samples.append(image_np[y:y+h, x:x+w])
                        ids.append(id)

        return face_samples, ids

    def train(self):
        """Train the face recognition model."""
        dataset_path = os.path.join(settings.MEDIA_ROOT, 'dataSet')
        face_samples, ids = self.get_images_and_labels(dataset_path)
        
        if len(face_samples) == 0 or len(ids) == 0:
            return False, "No training data found"
            
        try:
            self.model.train(face_samples, np.array(ids))
            model_path = os.path.join(settings.BASE_DIR, 'face_app/model/trainner.yml')
            self.model.save(model_path)
            return True, "Model trained successfully!"
        except Exception as e:
            return False, f"Error training model: {str(e)}" 