"""
Face recognition from static images.
"""
import cv2
import os
import numpy as np
from django.conf import settings
from .face_recognizer import FaceRecognizer

class FaceRecognitionFromImage:
    def __init__(self):
        self.face_recognizer = FaceRecognizer()
        self.face_detector = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'face_app/model/haarcascade_frontalface_default.xml'))
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontscale = 0.8
        self.fontcolor = (0, 255, 0)  # Green
        
    def process_image(self, image_path):
        """Process an image to detect and recognize faces."""
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                return None, "Error: Could not read the image."
            
            # Process with recognizer
            processed_img, results = self.face_recognizer.recognize_face_in_frame(img)
            
            # Save the processed image
            output_filename = os.path.basename(image_path)
            output_base, output_ext = os.path.splitext(output_filename)
            output_path = os.path.join(settings.MEDIA_ROOT, f"processed_{output_base}{output_ext}")
            cv2.imwrite(output_path, processed_img)
            
            # Return the relative path for web access
            relative_output_path = os.path.relpath(output_path, settings.MEDIA_ROOT)
            return f"{settings.MEDIA_URL}{relative_output_path}", results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error processing image: {str(e)}" 