"""
Face recognition functionality.
"""
import cv2
import numpy as np
import os
from django.conf import settings
from .definitions import NUM_MATCH_THRESHOLD, GREEN, BLUE

class FaceRecognizer:
    def __init__(self):
        # Initialize face detector
        self.face_detector = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'face_app/model/haarcascade_frontalface_default.xml'))
        if self.face_detector.empty():
            raise ValueError("Error: Haar Cascade file not found.")
        
        # Initialize face recognizer
        self.model = cv2.face.LBPHFaceRecognizer_create()
        model_path = os.path.join(settings.BASE_DIR, 'face_app/model/trainner.yml')
        
        if os.path.exists(model_path):
            try:
                self.model.read(model_path)
                self.model_loaded = True
            except Exception as e:
                self.model_loaded = False
                print(f"Error loading the model: {e}")
        else:
            self.model_loaded = False
            print("Model file not found. Train the model first.")
        
        # Set text style
        self.fontface = cv2.FONT_HERSHEY_SIMPLEX
        self.fontscale = 1
        self.fontcolor = GREEN
        self.fontcolor1 = BLUE

    def get_profile(self, id):
        """Get user information from database."""
        from face_app.models import Person
        try:
            person = Person.objects.get(id=id)
            return (person.id, person.name)
        except:
            return None

    def recognize_face_in_frame(self, frame):
        """Recognize faces in a frame."""
        if not self.model_loaded:
            return frame, [{'name': 'Model not trained', 'position': (10, 30)}]

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        
        results = []
        
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            try:
                # Recognize the face
                id, dist = self.model.predict(gray[y:y + h, x:x + w])
                
                profile = None
                if dist < NUM_MATCH_THRESHOLD:
                    profile = self.get_profile(id)
                
                # Add the name to the results
                if profile is not None:
                    name_text = f"Name: {profile[1]}"
                    cv2.putText(frame, name_text, (x, y + h + 30), self.fontface, self.fontscale, self.fontcolor, 2)
                    results.append({
                        'name': profile[1],
                        'position': (x, y),
                        'confidence': dist
                    })
                else:
                    cv2.putText(frame, "Name: Unknown", (x, y + h + 30), self.fontface, self.fontscale, self.fontcolor1, 2)
                    results.append({
                        'name': 'Unknown',
                        'position': (x, y),
                        'confidence': dist if 'dist' in locals() else None
                    })
            except Exception as e:
                print(f"Error in recognition: {e}")
                cv2.putText(frame, "Recognition error", (x, y + h + 30), self.fontface, self.fontscale, self.fontcolor1, 2)
                results.append({
                    'name': f"Error: {str(e)}",
                    'position': (x, y)
                })
        
        return frame, results

    def recognize_from_image(self, image_path):
        """Recognize faces from an image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            return None, "Error: Unable to read the image file."
        
        # Process the image
        processed_frame, results = self.recognize_face_in_frame(frame)
        
        # Save the processed image with recognition results
        output_path = image_path.replace('.', '_processed.')
        cv2.imwrite(output_path, processed_frame)
        
        return output_path, results 