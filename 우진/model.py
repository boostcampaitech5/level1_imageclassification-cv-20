import torch
import torch.nn as nn
import timm

class CustomedModel(nn.Module):
    def __init__(self, num_classes, pretrained = False) -> None:
        super().__init__()

        # resnet 
        self.model = timm.create_model('resnet50',num_classes = num_classes, pretrained = pretrained)


    def forward(self,x):
        return self.model(x)

    
# class CombinedModel(nn.Module):
    