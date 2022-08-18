import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision.transforms as trans




class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        super(MyDataset, self).__init__()
        self.image_path = path + '/' + 'horse.txt'
        self.label_path = path + '/' + 'mask.txt'
        self.path = path

        f = open(self.image_path, 'r')
        data_image = f.readlines()
        imgs = []
        for line in data_image:
            line = line.rstrip()
            imgs.append(os.path.join(self.path + '/horse', line))
        f.close()

        f2 = open(self.label_path, 'r')
        data_label = f2.readlines()
        labels = []
        for line in data_label:
            line = line.rstrip()
            labels.append(os.path.join(self.path + '/mask', line))
        f2.close()

        self.img = imgs
        self.label = labels
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]
        trans1 = trans.ToTensor()
        img = Image.open(img)
        # print(img.size())
        img = img.resize((224, 224))
        img = trans1(img)
        # print(img.size())

        # if self.transform is not None:
        #     img = self.transform(img)

        label = Image.open(label)
        label = label.resize((224, 224))
        label = trans1(label)*255

        return img, label

def make_txt(root, file_name):
    path = os.path.join(root, file_name)
    data = os.listdir(path)
    f = open(root+'/'+file_name+'.txt', 'w')
    for line in data:
        f.write(line+'\n')
    f.close()
    print('success')



def build_datasets(data_root,proportion):
    make_txt(data_root, file_name='horse')
    make_txt(data_root, file_name='mask')
    data = MyDataset(data_root, transform=None)
    train_size = int(len(data) * (1-proportion))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    return train_dataset, test_dataset

if  __name__ == '__main__':
    print(torch.__version__)