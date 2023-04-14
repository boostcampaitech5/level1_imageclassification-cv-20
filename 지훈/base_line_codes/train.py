import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor, Normalize, RandomAffine,RandomHorizontalFlip
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor, Normalize, RandomAffine,RandomHorizontalFlip
from torch.utils.tensorboard import SummaryWriter

from .model import CustomedModel, ClassModel
from .dataset import MaskMultiLabelDataset
from .evaluation import Evaluation

import pandas as pd
from sklearn.metrics import f1_score
import sys
import os
import datetime

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
def train_model(LEARNING_RATE=1e-3, EPOCH=5, BATCH_SIZE=64, PRETRAINED=False, LABEL='mask', VAL_SIZE=0.2):
    writer = SummaryWriter('logs/')
    DATA = pd.read_csv('train_data.csv')
    TRAIN_SIZE = 1 - VAL_SIZE
    # transform
    transform = transforms.Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])
    # model setting
    
    if LABEL == 'multi':
        NUM_CLASSES = 8
    elif LABEL == 'gender':
        NUM_CLASSES = 2
    else:
        NUM_CLASSES = 3

    # save dir and name
    if 'parameters' not in os.listdir('./'):
        os.mkdir('parameters')
    
    SAVE_DIR = './parameters/'
    SAVE_NAME = f'MaskModel'
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tssmsg = 'train setting start'
    len_tssmsg = len(tssmsg)
    tssmsg = '-'*((60-len_tssmsg)//2) + tssmsg + '-'*((60-len_tssmsg)//2)
    ######
    print(tssmsg)
    ######

    model = CustomedModel(num_classes = NUM_CLASSES).to(device)
    print('|---set model')
    criterion = nn.CrossEntropyLoss()
    print('|---set loss')
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    print('|---set optimizer')
    dataset = MaskMultiLabelDataset(DATA, transform = transform, label_option=LABEL) 
    print('|---set dataset')
    train_dataset, val_dataset = random_split(dataset, [TRAIN_SIZE,VAL_SIZE])
    loader, val_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE,shuffle = True), DataLoader(val_dataset, batch_size = BATCH_SIZE,shuffle = True)
    print('|---set dataloader')
    ######
    tsemsg = 'train setting end'
    len_tsemsg = len(tsemsg)
    tsemsg = '-'*((60-len_tsemsg)//2) + tsemsg + '-'*((60-len_tsemsg)//2)
    print(tsemsg)
    ######

    # train
    print('')
    print('train start')
    for epoch in range(1,EPOCH + 1):

        eval_by_epoch = Evaluation(num_classes = NUM_CLASSES)
        model.train()
        for iter, (image,label) in enumerate(loader):

            image = image.to(device)
            optimizer.zero_grad()
            pred = model(image)

            if LABEL == 'multi':
                print('multi-label 은 현재 train 까지만 구현이 되어있습니다.')
                gender_label, age_label, mask_label = label
                
                gender_label = torch.tensor(gender_label).to(device)
                age_label = torch.tensor(age_label).to(device)
                mask_label = torch.tensor(mask_label).to(device)
                total_label = gender_label + age_label + mask_label
                (gender_pred, age_pred, mask_pred) = torch.split(pred,[2,3,3],dim=1)

                gender_loss = criterion(gender_pred,gender_label)
                age_loss = criterion(age_pred,age_label)
                mask_loss = criterion(mask_pred,mask_label)
                
                # loss 가중치도 생각해보기 
                loss = gender_loss + age_loss + mask_loss
                loss.backward()
                optimizer.step()
                
                acc = eval_by_epoch.update(pred,torch.tensor(total_label))
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
                    print("gender acc %.4f | age acc %.4f | mask acc %.4f | total acc %.4f"%(train_gender_acc, train_age_acc, train_mask_acc, train_acc))


            else:
                label = label.clone().detach().to(device)
                loss = criterion(pred,label)
                loss.backward()
                optimizer.step()


                train_acc = eval_by_epoch.update(pred,label)
                # loss, accuracy 출력하기   
                if iter % 20 == 0:
                    train_loss = loss.item()
                    print("Iter [%3d/%3d] | Train Loss %.4f | Train Acc %.4f" %(iter, len(loader), train_loss, train_acc))

        label_acc, total_acc, f1_score = eval_by_epoch.result()
        print(f'label acc : {label_acc}, total_acc : {total_acc}, f1_score : {f1_score}')
        
        if LABEL == 'multi':
                writer.add_scalar(f"Train/Loss/Gender", gender_loss, epoch)
                writer.add_scalar(f"Train/Loss/Age", age_loss, epoch)
                writer.add_scalar(f"Train/Loss/Mask", mask_loss, epoch)
                writer.add_scalar(f"Train/Loss/Class", loss, epoch)
                
                writer.add_scalar(f"Train/Acc/Gender", train_gender_acc, epoch)
                writer.add_scalar(f"Train/Acc/Age", train_age_acc, epoch)
                writer.add_scalar(f"Train/Acc/Mask", train_mask_acc, epoch)
                writer.add_scalar(f"Train/Acc/Class", train_acc, epoch)
                
                writer.add_scalar(f"Train/F1_score", f1_score, epoch)
        else:
                writer.add_scalar(f"Train/Loss/{LABEL}", loss, epoch)
                
                writer.add_scalar(f"Train/Acc/{LABEL}", train_acc, epoch)
                
                writer.add_scalar(f"Train/F1_score/{LABEL}", f1_score, epoch)
                
        # validation_acc
        if LABEL != 'multi':
            model.eval()
            valid_loss, valid_acc = AverageMeter(), AverageMeter()
            eval_val = Evaluation(num_classes=NUM_CLASSES)
            for img, label in val_loader:
                # Validation에 사용하기 위한 image, label 처리 (필요한 경우, data type도 변경해주세요)
                img, label = img.float().cuda(), label.clone().detach().cuda()

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
            print("Valid Loss %.4f | Valid Acc %.4f"%(valid_loss, valid_acc))
            print(eval_val.result())


        print(f'{LABEL} epoch {epoch} is done!')   


        # 파라미터 저장하기
        if LABEL == 'multi':torch.save(model.state_dict(), SAVE_DIR + SAVE_NAME + "%d|loss_%.4f|gender_%.4f|age_%.4f|mask_%.4f|total_acc _%.4f.pt"%(epoch, train_loss, train_gender_acc,train_age_acc, train_mask_acc,train_acc))
        else : torch.save(model.state_dict(), SAVE_DIR + SAVE_NAME + "_%d_%.4f_%.4f.pt"%(epoch, train_loss, train_acc))


    print(f"{LABEL} train done")
    return model

def classify_label(MASK_MODEL, GENDER_MODEL, AGE_MODEL, BATCH_SIZE=64, VAL_SIZE=0.2):
    DATA = pd.read_csv('train_data.csv')
    TRAIN_SIZE = 1 - VAL_SIZE
    
    if 'parameters' not in os.listdir('./'):
        os.mkdir('parameters')
    
    SAVE_DIR = './parameters/'
    SAVE_NAME = f'MaskModel'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tssmsg = 'train setting start'
    len_tssmsg = len(tssmsg)
    tssmsg = '-'*((60-len_tssmsg)//2) + tssmsg + '-'*((60-len_tssmsg)//2)
    print(tssmsg)
    
    model = ClassModel(MASK_MODEL, GENDER_MODEL, AGE_MODEL).to(device)
    print('|---set model')
    dataset = MaskMultiLabelDataset(DATA, label_option='multi') 
    print('|---set dataset')
    loader= DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
    print('|---set dataloader')
    
    tsemsg = 'train setting end'
    len_tsemsg = len(tsemsg)
    tsemsg = '-'*((60-len_tsemsg)//2) + tsemsg + '-'*((60-len_tsemsg)//2)
    print(tsemsg)
    
    model.eval()
    valid_acc = AverageMeter()
    eval_val = Evaluation(num_classes=18)
    
    for img, label in loader:
        # Validation에 사용하기 위한 image, label 처리 (필요한 경우, data type도 변경해주세요)
        img, label = img.float().cuda(), label.clone().detach().cuda()

        # 모델에 이미지 forward (gradient 계산 X)
        with torch.no_grad():
            pred_logit = model(img)

        # Accuracy 계산
        pred_label = torch.argmax(pred_logit, 1)
        acc = (pred_label == label).sum().item() / len(img)

        valid_acc.update(acc, len(img))
        eval_val.update(pred_logit,label)

    valid_acc = valid_acc.avg

    print('VALIDATION')
    print("Valid Acc %.4f"%(valid_acc))
    print(eval_val.result())