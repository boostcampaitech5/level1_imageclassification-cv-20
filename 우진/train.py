import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor, Normalize, RandomAffine,RandomHorizontalFlip

from model import CustomedModel
from Dataset import MaskMultiLabelDataset
from evaluation import Evaluation

import pandas as pd
from sklearn.metrics import f1_score
import sys


# Misc
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

## hypter paramter setting 
LEARNING_RATE = 1e-3
EPOCH = 5
BATCH_SIZE = 64
RESIZE = False
PRETRAINED = False

# data paramter setting 
LABEL = 'age'
DATA = pd.read_csv('customed_train_data.csv')
VAL_SIZE = 0.2
TRAIN_SIZE = 1 - VAL_SIZE


# transform
transforms = transforms.Compose([
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])

# model setting 
NUM_CLASSES = 3

# save dir and name
SAVE_DIR = './parameters/'
SAVE_NAME = 'RNN_multi_age2'

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######
print('train setting start')
######

model = CustomedModel(num_classes = NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
dataset = MaskMultiLabelDataset(DATA, transform = transforms, label_option=LABEL) 
train_dataset, val_dataset = random_split(dataset, [TRAIN_SIZE,VAL_SIZE])
loader, val_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE,shuffle = True), DataLoader(val_dataset, batch_size = BATCH_SIZE,shuffle = True)




######
print('train setting end')
######
                                                    
# train

print('train start')
for epoch in range(1,EPOCH + 1):
    
    eval_by_epoch = Evaluation(num_classes = NUM_CLASSES)
    model.train()
    for iter, (image,label) in enumerate(loader):
        
        image = image.to(device)
        optimizer.zero_grad()
        pred = model(image)

        
        if LABEL == 'multi':
            gender_label, age_label, mask_label = label 
            gender_label = gender_label.to(device)
            age_label = age_label.to(device)
            mask_label = mask_label.to(device)
            
            (gender_pred, age_pred, mask_pred) = torch.split(pred,[2,3,3],dim=1)

            gender_loss = criterion(gender_pred,gender_label)
            age_loss = criterion(age_pred,age_label)
            mask_loss = criterion(mask_pred,mask_label)
            
            # loss 가중치도 생각해보기 
            loss = gender_loss + age_loss + mask_loss
            loss.backward()
            optimizer.step()
            
            # loss, accuracy 출력하기 
            if iter % 20 == 0:
                train_loss = loss.item()
                
                pred_gender_label = torch.argmax(gender_pred,1)
                pred_age_label = torch.argmax(age_pred,1)
                pred_mask_label = torch.argmax(mask_pred,1)
    
                train_gender_acc = (pred_gender_label == gender_label).sum().item() / len(image)
                train_age_acc = (pred_age_label == age_label).sum().item() / len(image)
                train_mask_acc = (pred_mask_label == mask_label).sum().item() / len(image)
                train_acc = ((pred_gender_label == gender_label) & (pred_age_label == age_label) & (pred_mask_label == mask_label)).sum().item() / len(image)
                
                print("Iter [%3d/%3d] | Train Loss %.4f" %(iter, len(loader), train_loss))
                print("gender acc %.4f | age acc %.4f | mask acc %.4f| total acc %.4f"%(train_gender_acc, train_age_acc, train_mask_acc, train_acc))
                
    
        else:
            label = label.to(device)
            loss = criterion(pred,label)
            loss.backward()
            optimizer.step()
            
            
            acc = eval_by_epoch.update(pred,label)
            # loss, accuracy 출력하기   
            if iter % 20 == 0:
                train_loss = loss.item()
                print("Iter [%3d/%3d] | Train Loss %.4f | Train Acc %.4f" %(iter, len(loader), train_loss, acc))

    label_acc, total_acc, f1_score = eval_by_epoch.result()
    print(f'label acc : {label_acc}, total_acc : {total_acc}, f1_score : {f1_score}')
    
    # validation_acc
    if True:
        model.eval()
        valid_loss, valid_acc = AverageMeter(), AverageMeter()
        eval_val = Evaluation(num_classes=NUM_CLASSES)
        for img, label in val_loader:
            # Validation에 사용하기 위한 image, label 처리 (필요한 경우, data type도 변경해주세요)
            img, label = img.float().cuda(), label.long().cuda()

            # 모델에 이미지 forward (gradient 계산 X)
            with torch.no_grad():
                pred_logit = model(img)

            # loss 값 계산
            loss = criterion(pred_logit, label)

            # Accuracy 계산
            pred_label = torch.argmax(pred_logit, 1)
            acc = (pred_label == label).sum().item() / len(img)

            valid_loss.update(loss.item(), len(img))
            valid_acc.update(acc, len(img))
            eval_val.update(pred_logit,label)

        valid_loss = valid_loss.avg
        valid_acc = valid_acc.avg
        
        # f1_score
        print('VALIDATION')
        print("Iter [%3d/%3d]| Valid Loss %.4f | Valid Acc %.4f"%(iter, len(val_loader), valid_loss, valid_acc))
        print(eval_val.result())
 
        
    print(f'epoch {epoch} is done!')   
    
    
    # 파라미터 저장하기
    if LABEL == 'multi':torch.save(model.state_dict(), SAVE_DIR + SAVE_NAME + "%d|loss_%.4f|gender_%.4f|age_%.4f|mask_%.4f|total_acc _%.4f.pt"%(epoch, train_loss, train_gender_acc,train_age_acc, train_mask_acc,train_acc))
    else : torch.save(model.state_dict(), SAVE_DIR + SAVE_NAME + "_%d_%.4f_%.4f.pt"%(epoch, train_loss, train_acc))
    
    
print('train done')