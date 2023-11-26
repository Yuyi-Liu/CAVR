import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import torch

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(256, num_class)
        self.fc3 = nn.Linear(512,num_class)

    # def forward(self, xyz1,xyz2):
    #     B, _, _ = xyz1.shape
        
    #     norm = None
    #     l1_xyz1, l1_points1 = self.sa1(xyz1, norm)
    #     l2_xyz1, l2_points1 = self.sa2(l1_xyz1, l1_points1)
    #     l3_xyz1, l3_points1 = self.sa3(l2_xyz1, l2_points1)
    #     x1 = l3_points1.view(B, 1024)
    #     x1 = self.drop1(F.relu(self.bn1(self.fc1(x1))))
    #     x1 = self.drop2(F.relu(self.bn2(self.fc2(x1))))

    #     l1_xyz2, l1_points2 = self.sa1(xyz2, norm)
    #     l2_xyz2, l2_points2 = self.sa2(l1_xyz2, l1_points2)
    #     l3_xyz2, l3_points2 = self.sa3(l2_xyz2, l2_points2)
    #     x2 = l3_points2.view(B, 1024)
    #     x2 = self.drop1(F.relu(self.bn1(self.fc1(x2))))
    #     x2 = self.drop2(F.relu(self.bn2(self.fc2(x2))))

    #     x = torch.cat((x1,x2),dim=-1)

    #     x = self.fc3(x)
    #     x = F.log_softmax(x, -1)


    #     return x,x1,x2

    def forward(self, xyz1):
        B, _, _ = xyz1.shape
        
        norm = None
        l1_xyz1, l1_points1 = self.sa1(xyz1, norm)
        l2_xyz1, l2_points1 = self.sa2(l1_xyz1, l1_points1)
        l3_xyz1, l3_points1 = self.sa3(l2_xyz1, l2_points1)
        x1 = l3_points1.view(B, 1024)
        x1 = self.drop1(F.relu(self.bn1(self.fc1(x1))))
        x1 = self.drop2(F.relu(self.bn2(self.fc2(x1))))



        return x1


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


