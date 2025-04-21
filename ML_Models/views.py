# /*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# All View Function
# Project:    Skin Disease Detection System
# Copyright:  Ratna Akter
# Developer:  Ratna Akter(Lead Full-Stack Developer)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

import os
import cv2
import numpy as np
from datetime import datetime
from django.conf import settings
from .models import UploadedImage
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from .ml_models import models as ml_models
import json

# This function preprocesses an input image by enhancing contrast using CLAHE, 
# converting color spaces, and applying Gaussian blur to reduce noise.
def preprocess_image(image):
    img = image.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    updated_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)
    denoised = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
    return denoised


def index(request):
    return render(request, 'index.html')


def ImageUploadForPredict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Save uploaded image
            skin_image = UploadedImage(image=request.FILES['image'])
            skin_image.save()

            # Get image path
            img_path = os.path.join(settings.MEDIA_ROOT, skin_image.image.name)

            # Preprocess the image for display
            original_img = cv2.imread(img_path)
            preprocessed_img = preprocess_image(original_img)

            # Save preprocessed image
            preprocessed_path = os.path.join(
                settings.MEDIA_ROOT, f"preprocessed_{skin_image.image.name}")
            cv2.imwrite(preprocessed_path, preprocessed_img)

            # Get model predictions
            results = []
            for model_name, model in ml_models.items():
                result = model.predict(img_path)
                results.append({
                    "name": model_name,
                    "prediction": result['prediction'],
                    "confidence": float(result['confidence']),
                    "metrics": {
                        key: float(value) for key, value in result['model_metrics'].items()
                    }
                })

            # Determine best model using a weighted score
            def calculate_model_score(model_result):
                metrics = model_result['metrics']
                # Weighted score with more emphasis on accuracy and F1 score
                return (
                    0.35 * metrics['accuracy'] +
                    0.30 * metrics['f1_score'] +
                    0.20 * metrics['recall'] +
                    0.15 * metrics['precision']
                )

            # Sort models by score
            sorted_models = sorted(
                results, key=calculate_model_score, reverse=True)
            best_model = sorted_models[0]

            # Also include the confidence in the best model selection
            final_scores = []
            for model in results:
                score = calculate_model_score(model)
                adjusted_score = (0.7 * score) + (0.3 * model['confidence'])
                final_scores.append(adjusted_score)

            best_index = np.argmax(final_scores)
            best_model = results[best_index]

            # Save prediction result
            skin_image.prediction = best_model['prediction']
            skin_image.save()

            # Create response data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response_data = {
                'image_url': skin_image.image.url,
                'preprocessed_url': f"/media/preprocessed_{skin_image.image.name}",
                'timestamp': timestamp,
                'models': results,
                'best_model': best_model,
                'success': True
            }
            print("response_data", response_data)

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse(response_data)

            return render(request, 'results.html', {
                'image': skin_image,
                'result': response_data
            })

        except Exception as e:
            error_data = {
                'success': False,
                'error': str(e)
            }
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse(error_data)
            return render(request, 'index.html', {'error': str(e)})

    return render(request, 'index.html')


def SingleModelPredict(request):
    if request.method == 'POST' and request.FILES.get('image') and request.POST.get('model_name'):
        try:
            # Get selected model
            model_name = request.POST.get('model_name')

            # Save uploaded image
            skin_image = UploadedImage(image=request.FILES['image'])
            skin_image.save()

            # Get image path
            img_path = os.path.join(settings.MEDIA_ROOT, skin_image.image.name)

            # Preprocess the image for display
            original_img = cv2.imread(img_path)
            preprocessed_img = preprocess_image(original_img)

            # Save preprocessed image
            preprocessed_path = os.path.join(
                settings.MEDIA_ROOT, f"preprocessed_{skin_image.image.name}")
            cv2.imwrite(preprocessed_path, preprocessed_img)

            # Get selected model prediction
            model = ml_models[model_name]
            result = model.predict(img_path)

            model_result = {
                "name": model_name,
                "prediction": result['prediction'],
                "confidence": float(result['confidence']),
                "metrics": {
                    key: float(value) for key, value in result['model_metrics'].items()
                }
            }

            # Save prediction result
            skin_image.prediction = model_result['prediction']
            skin_image.save()

            # Create response data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response_data = {
                'image_url': skin_image.image.url,
                'preprocessed_url': f"/media/preprocessed_{skin_image.image.name}",
                'timestamp': timestamp,
                'model': model_result,
                'success': True
            }

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse(response_data)

            return render(request, 'results.html', {
                'image': skin_image,
                'result': response_data
            })

        except Exception as e:
            error_data = {
                'success': False,
                'error': str(e)
            }
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse(error_data)
            return render(request, 'index.html', {'error': str(e)})

    return render(request, 'index.html')
