#!/usr/bin/env python3
"""
Demo Model Creation Script
Creates a simple pre-trained model for immediate testing
"""

import torch
import torch.nn as nn
from torchvision import models
import os

def create_demo_model():
    """Create a demo model for immediate testing"""
    print("🎯 Creating Demo Model for Plastic Classification")
    print("=" * 50)
    
    # Create a simple CNN model
    class DemoPlasticClassifier(nn.Module):
        def __init__(self, num_classes=6):
            super(DemoPlasticClassifier, self).__init__()
            
            # Use a smaller pre-trained model
            self.backbone = models.mobilenet_v2(pretrained=True)
            
            # Modify the classifier
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        
        def forward(self, x):
            return self.backbone(x)
    
    # Create and save model
    model = DemoPlasticClassifier()
    
    # Initialize with some reasonable weights (simulating training)
    for param in model.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    # Save the model
    torch.save(model.state_dict(), 'best_plastic_model.pth')
    
    print("✅ Demo model created and saved as 'best_plastic_model.pth'")
    print("🎉 You can now run the application for testing!")
    
    return model

if __name__ == "__main__":
    create_demo_model() 