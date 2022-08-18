import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
        # aa = input.cpu().detach().numpy()
        # bb = target.cpu().detach().numpy()
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


def iou_mean(pred, target, n_classes=1):
    ious = []
    iousSum = 0
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)
    pred = F.softmax(pred, dim=0)
    # cc = pred.cpu().detach().numpy()
    pred = torch.where(pred[1] > 0.5, 1, 0)
    # pred = torch.from_numpy(pred)
    # aa = pred.cpu().detach().numpy()
    pred = pred.view(-1)
    # target = np.array(target)
    # target = torch.from_numpy(target)
    target = target
    target = torch.tensor(target, dtype=torch.int64)
    # bb = target.cpu().detach().numpy()
    target = target.view(-1)

    for cls in range(1, n_classes + 1):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum / n_classes


def one_hot_changer(tensor, vector_dim, dim=-1):

    if tensor.dtype != torch.long:
        tensor = tensor.long()

    one_hot = torch.eye(vector_dim, device=tensor.device)
    vector = one_hot[tensor]

    dim_change_list = list(range(tensor.dim()))
    if dim == -1:
        return vector
    if dim < 0:
        dim += 1

    dim_change_list.insert(dim, tensor.dim())
    vector = vector.permute(dim_change_list)
    return vector.view(1,2,224,224)


class Edge_IoULoss(nn.Module):
    def __init__(self, n_class, edge_range=3, lamda=1.0, weight=None):
        super().__init__()
        self.n_class = n_class

        self.avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)

        if weight is None:
            self.weight = torch.ones([self.n_class])
        else:
            self.weight = weight
        self.lamda = lamda

    def edge_decision(self, seg_map):
        smooth_map = self.avgPool(seg_map)

        object_edge_inside_flag = seg_map * (smooth_map != seg_map)
        return object_edge_inside_flag

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        targets = one_hot_changer(targets, self.n_class, dim=1)
        predicts_idx = outputs.argmax(dim=1)
        predicts_seg_map = one_hot_changer(predicts_idx, self.n_class, dim=1)
        predict_edge = self.edge_decision(predicts_seg_map)
        targets_edge = self.edge_decision(targets)
        outputs = outputs * predict_edge
        targets = targets * targets_edge
        intersectoin = (outputs * targets).sum(dim=(2, 3))
        union = targets.sum(dim=(2, 3))
        edge_IoU_loss = (intersectoin + 1e-24) / (union + 1e-24)

        return  edge_IoU_loss



def BoundaryIoU(pred, target, n_classes=2):
    BoundaryIoULoss = Edge_IoULoss(n_class = n_classes)
    loss = BoundaryIoULoss(pred,target)
    loss = loss.cpu().detach().numpy().tolist()
    return  float(loss[0][0])

