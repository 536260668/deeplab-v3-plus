"""
Created on July 21 2022
@author: Liu Ziheng
"""

import argparse
import datetime
import random
import time
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from models.loss import DiceLoss, iou_mean,BoundaryIoU
from models.deeplabv3_plus import deeplabv3_plus

from make_dataset.build import build_datasets
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings('ignore')
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training Deeplabv3_plus', add_help=False)
    # SGD使用1e-2开始训练
    # ADAM使用1e-4开始训练
    parser.add_argument('--lr',default = 1e-4,type=float)
    parser.add_argument('--lr_backbone', default=1e-2,type=float)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--os', default=16, type=int)
    parser.add_argument('--freeze_bn', default=False, type=bool)
    # backbone选择
    parser.add_argument('--backbone', default='xception', type=str,
                        help="Name of the convolutional backbone to use(xception,resnet)")
    parser.add_argument('--pretrained', default=False, type=bool)
    parser.add_argument('--freeze_backbone', default=False, type=bool)
    # 随机种子选择
    parser.add_argument('--seed',default=19,type=int,
                        help='set the seed for reproducing the result')

    # 数据
    parser.add_argument('--data_root', default='./weizmann_horse_db',
                        help='path where to get the data')
    parser.add_argument('--testset_proportion', default=0.15,type=float)


    # 结果保存
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    # 运行选项
    parser.add_argument('--eval_freq',default = 5,type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--resume',help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
    parser.add_argument('--num_workers', default=0, type=int)

    return parser
def main(args):

    print(torch.__version__)
    print(torch.cuda.is_available())
    torch.multiprocessing.set_sharing_strategy('file_system')
    # setting the gui_id
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    # create the logging file
    filename_tail = args.backbone + '_' + args.optimizer
    run_log_name = 'run_log_'+filename_tail+'.txt'
    run_log_name = os.path.join(args.output_dir,run_log_name)
    with open(run_log_name,'a') as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))
    print(args)
    with open(run_log_name,'a') as log_file:
        log_file.write("{}".format(args))
    # setting gpu training
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    # criterion = DiceLoss()
    criterion = nn.CrossEntropyLoss()
    model = deeplabv3_plus(2,3,args.backbone,args.pretrained,args.os,args.freeze_bn,args.freeze_backbone)
    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    t = model_without_ddp.named_parameters()
    # use different optimation params for different parts of the model
    # 使用p for n, p in model_without_ddp.named_parameters()读取参数名字与参数值，对于backbone设定一定的参数，且要其要grad
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # SGD is used by default
    if args.optimizer=='SGD' or args.optimizer==None:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr)
    if args.optimizer=='Adam':
        optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # create the dataset

    # 得到训练集和验证集
    train_set, test_set = build_datasets(args.data_root,args.testset_proportion)

    # the dataloader for training
    data_loader_train = DataLoader(train_set,num_workers=args.num_workers,drop_last = True,batch_size=args.batch_size)

    data_loader_test = DataLoader(test_set, batch_size=1, drop_last=False, num_workers=args.num_workers)

    # for (image, label) in DataLoader:
    #     print(image.size(), label.size())


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        #args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    # save the performance during the training
    writer = SummaryWriter(args.tensorboard_dir)

    loss_test_list = []
    loss_epoch = []
    step = 0
    writer = SummaryWriter(args.tensorboard_dir)
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        running_loss = 0
        n = 0
        model.train()
        for index,data in enumerate(data_loader_train):
            n = n+1
            input = data[0]
            label = data[1]
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            loss = criterion(output,torch.squeeze(label).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_epoch.append(running_loss/n)
        t2 = time.time()
        if writer is not None:
            with open(run_log_name,'a') as log_file:
                log_file.write("{} epoch loss:{:.5f}, cost time: {:.2f}s \n".format(epoch,loss_epoch[epoch],t2-t1))
            writer.add_scalar('loss/loss',loss_epoch[epoch], epoch)
        print("{} epoch loss:{:.5f}, cost time: {} s ".format(epoch,loss_epoch[epoch],t2-t1))
        lr_scheduler.step()
        checkpoint_latest_path = 'latest_' + filename_tail + '.pth'
        checkpoint_latest_path = os.path.join(args.checkpoints_dir,checkpoint_latest_path)
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch
        }, checkpoint_latest_path)

        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            model.eval()
            loss_test = 0
            n = 0
            miou = 0
            biou = 0
            with torch.no_grad():
                for index, data in enumerate(data_loader_test):
                    n = n+1
                    input = data[0]
                    label = data[1]
                    input = input.to(device)
                    label = label.to(device)
                    output = model(input)
                    miou_item = iou_mean(output,label)
                    biou_item = BoundaryIoU(output,label)
                    loss = criterion(output, torch.squeeze(label,1).long())
                    loss_test += loss
                    miou += miou_item
                    biou +=biou_item
            miou = miou/n
            biou = biou/n
            loss_test = loss_test/n
            loss_test = loss_test.item()
            loss_test_list.append(loss_test)
            t2 = time.time()
            print('=======================================test=======================================')
            print("loss_test:", loss_test,"miou:",miou,"biou:",biou,"time:", t2 - t1, "min loss_test:", min(loss_test_list))
            with open(run_log_name, "a") as log_file:
                log_file.write("loss_test:{},miou:{},biou:{},time:{}, min loss_test:{}\n".format(loss_test,miou,biou,t2 - t1, min(loss_test_list)))
            print('=======================================test=======================================')
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("loss@{}: {},miou@{},miou@{}: {}\n".format(step, loss_test,step,miou,biou))
                writer.add_scalar('loss', loss_test, step)
                writer.add_scalar('miou', miou, step)
                writer.add_scalar('biou', biou, step)
                step += 1

            # save the bese model since begining
            if abs(min(loss_test_list)-loss_test)<0.01:
                checkpoint_latest_path = 'best_loss_' + filename_tail + '.pth'
                checkpoint_best_path = os.path.join(args.checkpoints_dir, checkpoint_latest_path)
                torch.save({
                    'model': model.state_dict(),
                }, checkpoint_best_path)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
