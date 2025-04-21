from django.db import models


class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=100, blank=True)

    def __str__(self) -> str:
        return "tested image"
