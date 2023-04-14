import os
import pandas as pd
import numpy as np

# code 폴더안에서 실행해주세요
def classify_data(gender, age, mask):
    class_num = 0
    
    # gender
    class_num += gender*3
    
    # age
    class_num += age
   
    # mask
    class_num += mask*6
    
    return class_num


def label_maker():
    train_dir = '/opt/ml/input/data/train'

    train_data_dict = {
                       'id': [],
                       'gender': [],
                       'age': [],
                       'mask': [],
                       'class_id': [],
                       'path': []
                      }
    image_id = 1

    for folder in os.listdir(train_dir + '/images'):
        if folder[0] == '.':
            continue
        image_gender = folder.split('_')[1]
        if image_gender == 'female':
            image_gender = 0
        else:
            image_gender = 1

        image_age = int(folder.split('_')[3])
        if image_age < 30:
            image_age = 0
        elif 30 <= image_age < 60:
            image_age = 1
        else:
            image_age = 2

        image_path = train_dir + '/images/' + folder

        for image in os.listdir(image_path):
            if image[0] == '.':
                continue

            # id
            train_data_dict['id'].append(image_id)
            image_id += 1
            # gender
            train_data_dict['gender'].append(image_gender)

            # age
            train_data_dict['age'].append(image_age)

            # mask
            if 'incorrect' in image:
                image_mask = 1
            elif 'mask' in image:
                image_mask = 0
            else:
                image_mask = 2

            train_data_dict['mask'].append(image_mask)

            # class
            train_data_dict['class_id'].append(classify_data(image_gender, image_age, image_mask))

            # path
            train_data_dict['path'].append(image_path + '/' + image)

    train_data_pd = pd.DataFrame(train_data_dict)
    print(f'데이터 수 : {len(train_data_pd)}')

    output_dir = '/opt/ml/code/'
    train_data_pd.to_csv(output_dir + 'train_data.csv', index=False)