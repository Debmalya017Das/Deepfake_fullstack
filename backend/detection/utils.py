# import os
# import numpy as np
# import torch
# import cv2
# import matplotlib.pyplot as plt
# from torchvision import models, transforms
# import torch.nn as nn

# # Constants
# IMG_SIZE = (224, 224)
# TIME_STEPS = 10
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class ResNetLSTM(nn.Module):
#     def __init__(self):
#         super(ResNetLSTM, self).__init__()
#         self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
#         self.resnet.fc = nn.Identity()  # Remove the fully connected layer
#         self.lstm = nn.LSTM(2048, 128, batch_first=True)
#         self.fc = nn.Linear(128, 1)

#     def forward(self, x):
#         batch_size, time_steps, C, H, W = x.size()
#         x = x.view(batch_size * time_steps, C, H, W)
#         features = self.resnet(x)
#         features = features.view(batch_size, time_steps, -1)
#         _, (hidden, _) = self.lstm(features)
#         out = self.fc(hidden[-1])
#         return torch.sigmoid(out)

# def preprocess_video(video_path):
#     """
#     Extracts and preprocesses frames from a video
#     """
#     frames = []
#     cap = cv2.VideoCapture(video_path)
    
#     # Extract frames
#     for _ in range(TIME_STEPS):
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Convert BGR to RGB
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = transforms.ToPILImage()(frame)
#         frame = transforms.Resize(IMG_SIZE)(frame)
#         frame = transforms.ToTensor()(frame)
#         frames.append(frame)
    
#     cap.release()
    
#     # Pad or truncate to TIME_STEPS
#     if len(frames) < TIME_STEPS:
#         while len(frames) < TIME_STEPS:
#             frames.append(torch.zeros(3, *IMG_SIZE))
#     else:
#         frames = frames[:TIME_STEPS]
    
#     # Normalize and prepare for model
#     transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     frames = torch.stack([transform(f) for f in frames])
    
#     return frames.unsqueeze(0)  # Add batch dimension

# def load_model(model_path):
#     """
#     Load pre-trained ResNet-LSTM model
#     """
#     model = ResNetLSTM()
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
#     model.eval()
#     return model
import os
import torch
import cv2
import numpy as np
from torchvision import models, transforms
import torch.nn as nn

# Constants
IMG_SIZE = (224, 224)
TIME_STEPS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=1024, num_layers=1):
        super(ResNetLSTM, self).__init__()
        # Use pretrained ResNet50
        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Remove the fully connected layer
        self.resnet.fc = nn.Identity()
        
        # LSTM layer with updated dimensions
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        
        # Reshape for ResNet feature extraction
        x = x.view(batch_size * time_steps, C, H, W)
        
        # Extract features using ResNet
        features = self.resnet(x)
        
        # Reshape features for LSTM
        features = features.view(batch_size, time_steps, -1)
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(features)
        
        # Use the last hidden state for classification
        out = self.fc(hidden[-1])
        
        return torch.sigmoid(out)

def preprocess_video(video_path):
    """
    Extracts and preprocesses frames from a video
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    # Extract frames
    for _ in range(TIME_STEPS):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transforms.ToPILImage()(frame)
        frame = transforms.Resize(IMG_SIZE)(frame)
        frame = transforms.ToTensor()(frame)
        frames.append(frame)
    
    cap.release()
    
    # Pad or truncate to TIME_STEPS
    if len(frames) < TIME_STEPS:
        while len(frames) < TIME_STEPS:
            frames.append(torch.zeros(3, *IMG_SIZE))
    else:
        frames = frames[:TIME_STEPS]
    
    # Normalize and prepare for model
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    frames = torch.stack([transform(f) for f in frames])
    
    return frames.unsqueeze(0)  # Add batch dimension

def load_model(model_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = ResNetLSTM()
    state_dict = torch.load(model_path, map_location=device)
    
    model_state_dict = model.state_dict()
    
    # Print out all keys in the saved state dict for comparison
    print("Saved model keys:", list(state_dict.keys()))
    print("Current model keys:", list(model_state_dict.keys()))
    
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if model_state_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"Shape mismatch for {k}: expected {model_state_dict[k].shape}, got {v.shape}")
                # Optional: try to reshape or handle the mismatch
        else:
            print(f"Skipping unexpected key: {k}")
    
    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model
