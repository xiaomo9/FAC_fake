import os
import torch
from torchvision import transforms, datasets
from augmentation import Aug
#Install TPU environment using this code    
#!curl install https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
#!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#DFDC dataset mean and standard
# mean = [0.4718, 0.3467, 0.3154]
# std = [0.1656, 0.1432, 0.1364]


data_transforms = {
    'train': transforms.Compose([
        Aug(),
        transforms.transforms.ColorJitter(brightness=(0.7, 1.3),
            contrast=(0.7, 1.3),
            saturation=(0.7, 1.3),
            hue=(-0.5, 0.5)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'validation': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# load data

def session(cession='g', data_dir = 'sample/', batch_size=32):
    batch_size=batch_size
    data_dir = data_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation', 'test']}
    
    if cession=='t':
        dataloaders, dataset_sizes = load_tpu(image_datasets, batch_size, data_dir)
        return batch_size, dataloaders, dataset_sizes
    else:
        dataloaders, dataset_sizes = load_gpu(image_datasets, batch_size, data_dir)
        return batch_size, dataloaders, dataset_sizes

def load_gpu(image_datasets, batch_size, data_dir):
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size,
                                                 shuffle=True, num_workers=4, pin_memory=True)
                   for x in ['train', 'validation', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    
    return dataloaders, dataset_sizes

