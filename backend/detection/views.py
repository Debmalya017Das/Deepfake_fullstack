import os
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import preprocess_video, load_model
from torch import torch

class DeepfakeDetectionView(APIView):
    def post(self, request):
        # Check if video is in request
        if 'video' not in request.FILES:
            return Response({'error': 'No video uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        
        video = request.FILES['video']
        
        # Save uploaded video
        video_path = os.path.join(settings.MEDIA_ROOT, 'uploads', video.name)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        with open(video_path, 'wb+') as destination:
            for chunk in video.chunks():
                destination.write(chunk)
        
        try:
            # Load model
            model_path = os.path.join(settings.BASE_DIR, 'models', 'best_deepfake_model.pth')
            model = load_model(model_path)
            
            # Preprocess video
            video_tensor = preprocess_video(video_path)
            
            # Predict
            with torch.no_grad():
                prediction = model(video_tensor)
                is_fake = prediction.item() > 0.5
            
            return Response({
                'prediction': prediction.item(),
                'is_fake': is_fake
            })
        
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            # Optional: Remove uploaded video after processing
            if os.path.exists(video_path):
                os.remove(video_path)
