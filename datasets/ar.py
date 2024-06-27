from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np

class Ar(iData):
    '''
    Dataset Name:   Skin8 (ISIC_2019_Classification)
    Task:           Skin disease classification
    Data Format:    600x450 color images.
    Data Amount:    3555 for training, 705 for validationg/testing
    Class Num:      8
    Notes:          balanced each sample num of each class

    Reference:      
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.has_valid = True
        self.img_size = 224 if img_size is None else img_size
        self.train_trsf = [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(224, (0.8, 1)),
            ]
        
        self.test_trsf = []
        
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.60298395, 0.4887822, 0.46266827], std=[0.25993535, 0.24081337, 0.24418062]),
        ]

        self.class_order = np.arange(9).tolist()
    
    def getdata(self, fn, img_dir):
        #fn:os.path.join("datasets", "AR_train.csv")
        #img_dir:os.environ["DATA"]
        print(fn)
        file = open(fn)
        file_name_list = file.read().split('\n')[1:-1]
        
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(',')
            # if len(temp)==53 and temp[0]!='':
            if len(temp)==10 and temp[0]!='':
                file_path = os.path.join(img_dir, temp[0])
                if os.path.exists(file_path):
                    data.append(file_path)
                    targets.append(np.array([int(i) for i in temp[1:]]))
            else:
                print(temp[0])
        return np.array(data), np.array(targets)

    def download_data(self):
        base_dir = os.environ["DATA"]
        train_dir = os.path.join(os.environ["DATA"], "AR_train.csv")
        val_dir = os.path.join(os.environ["DATA"], "AR_validation.csv")
        test_dir = os.path.join(os.environ["DATA"], "AR_test.csv")
        
        self.train_data, self.train_targets = self.getdata(train_dir, base_dir)
        self.valid_data, self.valid_targets = self.getdata(val_dir, base_dir)
        self.test_data, self.test_targets = self.getdata(test_dir, base_dir)