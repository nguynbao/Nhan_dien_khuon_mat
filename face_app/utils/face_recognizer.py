"""
Face recognition functionality with optimized performance.
"""
import cv2
import numpy as np
import os
from django.conf import settings
from django.core.cache import cache
from .definitions import NUM_MATCH_THRESHOLD, GREEN, BLUE

class FaceRecognizer:
    def __init__(self):
        # Initialize face detector with optimized parameters
        self.face_detector = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'model/haarcascade_frontalface_default.xml'))
        if self.face_detector.empty():
            raise ValueError("Error: Haar Cascade file not found.")
        
        # Initialize face recognizer
        self.model = cv2.face.LBPHFaceRecognizer_create(
            radius=1,  # Smaller radius for faster computation
            neighbors=8,
            grid_x=8,  # Optimized grid size
            grid_y=8,
            threshold=NUM_MATCH_THRESHOLD
        )
        model_path = os.path.join(settings.BASE_DIR, 'model/trainner.yml')
        
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
        
        # Cache for storing user profiles
        self.profile_cache = {}
        
        # Performance optimization parameters
        self.min_face_size = (30, 30)  # Minimum face size to detect
        self.scale_factor = 1.1  # Smaller scale factor for faster detection
        self.min_neighbors = 4  # Reduced for better performance
        
        # Frame processing optimization
        self.process_every_n_frames = 2  # Process every 2nd frame
        self.frame_counter = 0
        self.last_results = None

    def get_profile(self, id):
        """Get user information from database with caching."""
        # Check cache first
        cache_key = f'user_profile_{id}'
        
        # Try to get from memory cache
        if id in self.profile_cache:
            return self.profile_cache[id]
        
        # Try to get from Django cache
        cached_profile = cache.get(cache_key)
        if cached_profile:
            self.profile_cache[id] = cached_profile
            return cached_profile
        
        # If not in cache, get from database
        from face_app.models import Person
        try:
            person = Person.objects.get(id=id)
            profile = (person.id, person.name)
            
            # Store in both caches
            self.profile_cache[id] = profile
            cache.set(cache_key, profile, timeout=3600)  # Cache for 1 hour
            
            return profile
        except:
            return None

    def recognize_face_in_frame(self, frame):
        """Recognize faces in a frame with optimized performance."""
        if not self.model_loaded:
            return frame, [{'name': 'Model not trained', 'position': (10, 30)}]

        # Skip frames for performance
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n_frames != 0:
            if self.last_results:
                return frame, self.last_results
        
        # Resize frame for faster processing if too large
        height, width = frame.shape[:2]
        max_width = 640
        if width > max_width:
            scale = max_width / width
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optimize contrast for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces in the frame with optimized parameters
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            # Convert numpy types to Python native types for JSON serialization
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            try:
                # Extract face ROI and normalize
                face_roi = gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (100, 100))  # Standardize size
                
                # Recognize the face
                id, dist = self.model.predict(face_roi)
                # Convert numpy types to Python native types
                id, dist = int(id), float(dist)
                
                profile = None
                if dist < NUM_MATCH_THRESHOLD:
                    profile = self.get_profile(id)
                
                # Add the name to the results
                if profile is not None:
                    name_text = f"Name: {profile[1]}"
                    cv2.putText(frame, name_text, (x, y + h + 30), self.fontface, self.fontscale, self.fontcolor, 2)
                    results.append({
                        'name': profile[1],
                        'position': (int(x), int(y)),
                        'confidence': float(dist)
                    })
                else:
                    cv2.putText(frame, "Unknown", (x, y + h + 30), self.fontface, self.fontscale, self.fontcolor1, 2)
                    results.append({
                        'name': 'Unknown',
                        'position': (int(x), int(y)),
                        'confidence': float(dist) if 'dist' in locals() else None
                    })
            except Exception as e:
                print(f"Error in recognition: {e}")
                cv2.putText(frame, "Error", (x, y + h + 30), self.fontface, self.fontscale, self.fontcolor1, 2)
                results.append({
                    'name': f"Error: {str(e)}",
                    'position': (int(x), int(y))
                })
        
        self.last_results = results
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