import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from rearrange_on_proc.arguments import args
import importlib
import numpy as np
import torch
import pointnet.Pointnet2_pytorch.models.pointnet2_cls_msg as pointnet2_cls_msg
from rearrange_on_proc.utils.utils import cosine_similarity

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    point_num = point.shape[0]
    if point_num < npoint:
        new_point = np.zeros((npoint,3))
        new_point[:point_num,:] = point
        new_point[point_num:,:] = point[0,:]
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def preprocess(pc,npoint=1024):
    point_set = farthest_point_sample(pc,npoint)
    point_set = pc_normalize(point_set)
    return point_set

class PointNet2Helper():
    def __init__(self,device_id=1) -> None:
        self.device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
        # models_name = f"{os.path.dirname(os.path.realpath(__file__))}/Pointnet_Pointnet2_pytorch/models"
        # model_name = os.path.join(models_name,"pointnet2_cls_msg")
        # model = importlib.import_module(model_name)
        self.classifier = pointnet2_cls_msg.get_model(num_class = 2).to(self.device)
        checkpoint = torch.load(f"{os.path.dirname(os.path.realpath(__file__))}/best_model.pth",map_location=self.device)
        # checkpoint = torch.load("/home/lyy/rearrange_on_ProcTHOR/pointnet/Pointnet_Pointnet2_pytorch/log/classification/binary_msg_50/checkpoints/best_model.pth",map_location=self.device)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
    
    # def pointcloud_binary_cls(self,point_cloud1,point_cloud2):
    #     point_cloud1 = preprocess(point_cloud1)
    #     point_cloud2 = preprocess(point_cloud2)
    #     point_cloud1 = point_cloud1[np.newaxis,:]
    #     point_cloud2 = point_cloud2[np.newaxis,:]
    #     point_cloud1,point_cloud2 = torch.from_numpy(point_cloud1),torch.from_numpy(point_cloud2)
    #     point_cloud1 = point_cloud1.transpose(2,1).to(device=self.device,dtype=torch.float32)
    #     point_cloud2 = point_cloud2.transpose(2,1).to(device=self.device,dtype=torch.float32)
    #     with torch.no_grad():
    #         pred, x1,x2 = self.classifier(point_cloud1,point_cloud2)
    #     # print(pred)
    #     # pred = torch.exp(pred)[0][1]
    #     similarity = cosine_similarity(x1.cpu().squeeze().numpy(),x2.cpu().squeeze().numpy())
    #     # return pred.item()
    #     return similarity
    def pointcloud_binary_cls(self,point_cloud1):
        point_cloud1 = preprocess(point_cloud1)
        
        point_cloud1 = point_cloud1[np.newaxis,:]
        
        point_cloud1 = torch.from_numpy(point_cloud1)
        point_cloud1 = point_cloud1.transpose(2,1).to(device=self.device,dtype=torch.float32)
        
        with torch.no_grad():
            x1 = self.classifier(point_cloud1)
        # print(pred)
        # pred = torch.exp(pred)[0][1]
        # similarity = cosine_similarity(x1.cpu().squeeze().numpy(),x2.cpu().squeeze().numpy())
        # return pred.item()
        return x1.cpu().squeeze().numpy()

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    pointnet2_helper = PointNet2Helper()
    x = np.random.rand(2000,3)
    y = np.random.rand(2000,3)
    print(pointnet2_helper.pointcloud_binary_cls(x,y))
