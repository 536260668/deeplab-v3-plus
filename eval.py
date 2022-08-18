"""
Created on August 18 2022
@author: Liu Ziheng
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import cv2
import matplotlib.pyplot as plt

from models.loss import DiceLoss, iou_mean,BoundaryIoU
from models.deeplabv3_plus import deeplabv3_plus

from make_dataset.build import build_datasets

best_r = './ckpt/best_loss_resnet_Adam.pth'
best_x = './ckpt/best_loss_xception_Adam.pth'
model_r = deeplabv3_plus(2,3,'resnet',False,8,False,False)
model_x = deeplabv3_plus(2,3,'xception',False,16,False,False)
try:
    data = torch.load(best_r)
    model_r.load_state_dict(data['model'])
    data = torch.load(best_x)
    model_x.load_state_dict(data['model'])
    print("模型载入")
except FileNotFoundError:
    print("未载入模型")

seed = 19
torch.manual_seed(seed)
random.seed(seed)

train_set, test_set = build_datasets('./weizmann_horse_db',0.15)
data_loader_test = DataLoader(test_set, batch_size=1, drop_last=False)

device = torch.device('cuda')
criterion = nn.CrossEntropyLoss()
model_r.to(device)
model_r.eval()
model_x.to(device)
model_x.eval()
loss_test = 0
n = 0
miou = 0
biou = 0
with torch.no_grad():
    for index, data in enumerate(data_loader_test):
        n = n + 1
        input = data[0]
        label = data[1]
        input = input.to(device)
        label = label.to(device)
        output = model_r(input)
        miou_item = iou_mean(output, label)
        biou_item = BoundaryIoU(output, label)
        loss = criterion(output, torch.squeeze(label, 1).long())
        loss_test += loss
        miou += miou_item
        biou += biou_item
        if n % 5 == 0:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

            output2 = model_x(input)
            input = input.view(3,224,224)
            input = input.swapaxes(0, 1)
            input = input.swapaxes(1, 2)
            label = label.view(224, 224, 1)
            output = output.argmax(dim=1)
            output = output.view(224, 224, 1)
            output2 = output2.argmax(dim=1)
            output2 = output2.view(224, 224, 1)
            ax = plt.subplot(2, 2, 1)
            ax.set_title('原图')
            plt.imshow(input.cpu().detach().numpy())

            ax = plt.subplot(2, 2, 2)
            ax.set_title('label')
            plt.imshow(label.cpu().detach().numpy(), cmap='gray')

            ax = plt.subplot(2, 2, 3)
            ax.set_title('resnet pred')
            plt.imshow(output.cpu().detach().numpy(), cmap='gray')

            ax = plt.subplot(2, 2, 4)
            ax.set_title('Xception pred')
            plt.imshow(output2.cpu().detach().numpy(), cmap='gray')

            plt.show()

miou = miou / n
biou = biou / n
loss_test = loss_test / n
loss_test = loss_test.item()
print('=======================================resnet test=======================================')
print("loss_test:", loss_test, "miou:", miou, "biou:", biou)


loss_test = 0
n = 0
miou = 0
biou = 0
with torch.no_grad():
    for index, data in enumerate(data_loader_test):
        n = n + 1
        input = data[0]
        label = data[1]
        input = input.to(device)
        label = label.to(device)
        output = model_x(input)
        miou_item = iou_mean(output, label)
        biou_item = BoundaryIoU(output, label)
        loss = criterion(output, torch.squeeze(label, 1).long())
        loss_test += loss
        miou += miou_item
        biou += biou_item

miou = miou / n
biou = biou / n
loss_test = loss_test / n
loss_test = loss_test.item()
print('=======================================xception test=======================================')
print("loss_test:", loss_test, "miou:", miou, "biou:", biou)