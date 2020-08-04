import torch
import torchvision
import torch.utils.data
import random
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
import time

dataset = ImageFolder(
    "val",
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
)


num_workers = 0


print("[1]")
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32,num_workers=num_workers)
start = time.time()
for item, label in train_loader:
    print(item[0,0,112,112])
print("elapsed:",time.time()-start)



# ====================================================

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    # https://github.com/galatolofederico/pytorch-balanced-batch
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            return dataset.imgs[idx][1]

    def __len__(self):
        return self.balanced_max*len(self.keys)

print("[2]")
train_loader = torch.utils.data.DataLoader(dataset, sampler=BalancedBatchSampler(dataset), batch_size=32,num_workers=num_workers)
start = time.time()
for item, label in train_loader:
    label = label.cpu().numpy()
    result = np.where(label == 15)
    print(item[result[0],0,112,112])
print("elapsed:",time.time()-start)
