import torch
import sys, os
os.chdir(r'E:\Daima\face_fake\CViT-main')
import torch
import sys, os
import argparse
import numpy as np
import pandas as pd
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
import time
import copy
import pickle

sys.path.insert(1,'helpers')
sys.path.insert(1,'model')
sys.path.insert(1,'weight')

from augmentation import Aug
from cvit_GGCA_ADD_DEConv_RepBn8 import CViT
from loader import session
from torchsummary import summary

model = CViT(image_size=224, patch_size=7, num_classes=2, channels=512,
            dim=1024, depth=6, heads=8, mlp_dim=2048).cuda()

summary(model,(3,224,224))