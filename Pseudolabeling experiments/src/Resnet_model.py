from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import torch.nn as nn 

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential(
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
            nn.Linear(1000, 10))
            # nn.Softmax())
    def forward(self, input_data):    
        feature = self.feature(input_data) 
        return feature    
