from torch.utils.data import Dataset,DataLoader
from PIL import Image

class MaskMultiLabelDataset(Dataset):
    num_classes = 3 + 2 + 3
    def __init__(self,data,transform = None,label_option = 'one_hot') -> None:
        super().__init__()
        # input
        self.input_img_path = data['path']
        self.label_option = label_option
        # label
        if label_option == 'one_hot':
            self.label = data['class']
        elif label_option == 'multi':
            # gender, age, mask 하나로 뭉치기 
            self.gender_label = data['gender']
            self.age_label = data['age']
            self.mask_label = data['mask']
        elif label_option == 'gender':
            self.label = data['gender']
        elif label_option == 'age':
            self.label = data['age']
        elif label_option == 'mask':
            self.label = data['mask']
        else : assert True, 'label_option이 잘못되었습니다.'

        # transform
        self.transform = transform
    
    def __getitem__(self,index):
        # input img
        x = Image.open(self.input_img_path[index])

        # label
        if self.label_option == 'multi':
            gender = self.gender_label[index]
            age = self.age_label[index]
            mask = self.mask_label[index]
            y = (gender,age,mask)
        
        else : y = self.label[index]
        
        # transform
        if self.transform:
            x = self.transform(x)
        
        return x,y
    
    def __len__(self):
        return len(self.input_img_path)

# dataloader 만들기

# class MaskMultiLabelLoader(DataLoader):
    
    