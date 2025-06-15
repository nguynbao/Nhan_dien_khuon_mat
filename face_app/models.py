"""
Models for the face recognition app.
"""
from django.db import models
import os
from django.conf import settings

class Person(models.Model):
    """Model representing a person whose face will be recognized."""
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name
    
    def get_face_count(self):
        """Get the number of face images for this person."""
        dataset_folder = os.path.join(settings.MEDIA_ROOT, 'dataSet', f"{self.name}_{self.id}")
        if os.path.exists(dataset_folder):
            return len([f for f in os.listdir(dataset_folder) if f.endswith('.jpg')])
        return 0

class CapturedImage(models.Model):
    """Model representing a captured face image."""
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to='dataSet/')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Image of {self.person.name}"

class RecognizedFace(models.Model):
    """Model representing a recognized face."""
    original_image = models.ImageField(upload_to='recognized/original/')
    processed_image = models.ImageField(upload_to='recognized/processed/')
    person = models.ForeignKey(Person, on_delete=models.SET_NULL, null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    is_recognized = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        if self.is_recognized and self.person:
            return f"Recognized {self.person.name} ({self.confidence_score})"
        return "Unrecognized face"
