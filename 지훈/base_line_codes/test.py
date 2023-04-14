import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor, Normalize

from model import CustomedModel
from Dataset import MaskMultiLabelDataset

import pandas as pd
from sklearn.metrics import f1_score
import sys

LABEL = 'multi'
MODEL = './parameters/RNN_multilabel_pretrained5|loss_0.5857|gender_0.9844|age_0.7812|mask_0.9844|total_acc _0.7656.pt'
DATA = pd.read_csv('customed_train_data.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1

# model setting 
NUM_CLASSES = 8

# transform
transforms = transforms.Compose([
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    
])

model = CustomedModel(num_classes = NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL))
dataset = MaskMultiLabelDataset(DATA, transform = transforms, label_option=LABEL)
loader = DataLoader(dataset,batch_size = BATCH_SIZE,shuffle=True, drop_last=True)

# eval
model.eval()

train_gender_cnt, train_age_cnt, train_mask_cnt, train_acc_cnt = 0,0,0,0


# f1 score
pred_gender_labels, pred_age_labels, pred_mask_labels = [],[],[]
gender_labels,age_labels,mask_labels = [],[],[]
labels = []
preds = []


# total training accuracy
for iter, (image,label) in enumerate(loader):
    
    image = image.to(device)
    pred = model(image)
    

    if LABEL == 'multi':
        gender_label, age_label, mask_label = label 
        gender_label = gender_label.to(device)
        age_label = age_label.to(device)
        mask_label = mask_label.to(device)

        (gender_pred, age_pred, mask_pred) = torch.split(pred,[2,3,3],dim=1)
        
        pred_gender_label = torch.argmax(gender_pred,1)
        pred_age_label = torch.argmax(age_pred,1)
        pred_mask_label = torch.argmax(mask_pred,1)
        
        # 맞춘 개수 
        train_gender_cnt += (pred_gender_label == gender_label).sum().item()
        train_age_cnt += (pred_age_label == age_label).sum().item()
        train_mask_cnt += (pred_mask_label == mask_label).sum().item()
        train_acc_cnt += ((pred_gender_label == gender_label) & (pred_age_label == age_label) & (pred_mask_label == mask_label)).sum().item()
        
        # prediction 값 저장하기 
        pred_gender_labels.extend(pred_gender_label.tolist())
        pred_age_labels.extend(pred_age_label.tolist())
        pred_mask_labels.extend(pred_mask_label.tolist())
        
        # label 값 저장하기 
        gender_labels.extend(gender_label.tolist())
        age_labels.extend(age_label.tolist())
        mask_labels.extend(mask_label.tolist())
        
        
        
    else:
        label = label.to(device)
        pred_label = torch.argmax(pred,1)
        train_acc_cnt += (pred_label == label).sum().item()
        labels.extend(label.tolist())
        preds.extend(pred_label.tolist())
    
    if iter%1000 == 0: 
        print(f'iter {iter} is done!')       
        
        
        
        
# 결과 출력 
length = len(dataset)
if LABEL == 'multi':
    # f1 score 
    print(f1_score(pred_gender_labels,gender_labels,average='weighted'))
    print(f1_score(pred_age_labels,age_labels,average='weighted'))
    print(f1_score(pred_mask_labels,mask_labels,average='weighted'))
    
    #acc
    print(train_gender_cnt/length, train_age_cnt/length, train_mask_cnt / length, train_acc_cnt / length)
    
    
else : 
    #f1_score
    print(f1_score(labels,preds,average = 'weighted'))
    # acc
    print(train_acc_cnt / length)
    
print('testing is done')


# 각 라벨별로 얼마나 잘 맞추는지 ㅎ