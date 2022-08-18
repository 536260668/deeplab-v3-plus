import os

# # #resnet
# cmd = 'python train.py --lr 1e-4  --lr_backbone 1e-4  --lr_drop 40   --backbone "resnet" --freeze_backbone = True --os =16'
# os.system(cmd)
# cmd = 'python train.py --lr 1e-4  --lr_backbone 1e-4  --lr_drop 40   --backbone "resnet" --freeze_backbone = True --os = 8'
# os.system(cmd)
# cmd = 'python train.py --lr 1e-4  --lr_backbone 1e-4  --lr_drop 40   --backbone "resnet"  --freeze_backbone = False --os = 16'
# os.system(cmd)
# cmd = 'python train.py --lr 1e-4  --lr_backbone 1e-4  --lr_drop 40   --backbone "resnet"  --freeze_backbone = False --os = 8'
# os.system(cmd)
# #xception
# cmd = 'python train.py --lr 1e-4  --lr_backbone 1e-4  --lr_drop 40   --backbone "xception" --freeze_backbone = True --os =16'
# os.system(cmd)
# cmd = 'python train.py --lr 1e-4  --lr_backbone 1e-4  --lr_drop 40   --backbone "xception" --freeze_backbone = True --os = 8'
# os.system(cmd)
# cmd = 'python train.py --lr 1e-4  --lr_backbone 1e-4  --lr_drop 40   --backbone "xception"  --freeze_backbone = False --os = 16'
# os.system(cmd)
# cmd = 'python train.py --lr 1e-4  --lr_backbone 1e-4  --lr_drop 40   --backbone "xception"  --freeze_backbone = False --os = 8'
# os.system(cmd)

cmd = 'python train.py --freeze_backbone False --lr 1e-4  --lr_backbone 1e-4 --pretrained False --lr_drop 40   --backbone "xception" --freeze_backbone False --os 16 --pretrained False'
os.system(cmd)
cmd = 'python train.py --lr 1e-4  --lr_backbone 1e-4  --lr_drop 40   --backbone "resnet" --freeze_backbone False --os 16 --pretrained False'
os.system(cmd)