import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
import time

warnings.filterwarnings('ignore')


class lyyDataLoader(Dataset):
    def __init__(self,split):
        assert (split == 'train' or split == 'test')
        # if split == "train":
        self.root_path = "/home/lyy/rearrange_on_ProcTHOR/pointnet/datagen/processed_dataset0_50"
        # else:
        # self.root_path = "/home/lyy/rearrange_on_ProcTHOR/pointnet/datagen/processed_dataset100_110"
        paths_path = os.path.join(self.root_path,f"{split}_paths.txt")
        labels_path = os.path.join(self.root_path,f"{split}_labels.txt")
        file_path_list = [line.rstrip() for line in open(paths_path)]
        self.label_list = [int(line.rstrip()) for line in open(labels_path)]
        self.datapath = [os.path.join(self.root_path,file_path) for file_path in file_path_list]
        
        print("dataset size",len(self.datapath))
        

    def __len__(self):
        assert len(self.datapath) == len(self.label_list)
        return len(self.datapath)

    def _get_item(self, index):
        
        fn = self.datapath[index]
        cls = self.label_list[index]
        label = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn, delimiter=' ').astype(np.float32)
        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = lyyDataLoader(split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=24, shuffle=True)
    num = 0
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
        num+=1
    print(num)