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

class ClassModel(nn.Module):
    def __init__(self, mask_model, gender_model, age_model):
        super().__init__()
        
        self.mask = mask_model
        self.gender = gender_model
        self.age = age_model
        
    def forward(self, x):
        return self.mask(x)*6 + self.gender*3 + self.age
        