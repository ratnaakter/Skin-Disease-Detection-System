from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name='ML_Models'

urlpatterns = [
    path('',views.index, name='Home'),
    path('upload_view',views.ImageUploadForPredict, name='upload_view'),
    path('single-model-predict/', views.SingleModelPredict, name='single_model_predict'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
