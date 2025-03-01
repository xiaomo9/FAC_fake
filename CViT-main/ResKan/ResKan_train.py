import torch
import sys, os
os.chdir('/home/AIBike_ViG/interns/chenao/rknnkit/test/CViT-main')
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

sys.path.insert(1,'/home/AIBike_ViG/interns/chenao/rknnkit/test/CViT-main/helpers')
sys.path.insert(1,'/home/AIBike_ViG/interns/chenao/rknnkit/test/CViT-main/model')
sys.path.insert(1,'/home/AIBike_ViG/interns/chenao/rknnkit/test/CViT-main/weight')

from augmentation import Aug
from kan_resnet import resnet34
from loader import session
import optparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default
cession = 'g'  # GPU runtime
epoch = 140
dir_path = "/home/AIBike_ViG/interns/chenao/rknnkit/data/train_data"
# dir_path = './sample_train_data'
batch_size = 1024
min_val_loss = 0.6
min_val_acc = 0.67
lr = 0.00001
weight_decay = 0.001

parser = optparse.OptionParser("Train CViT model.")
parser.add_option("-e", "--epoch", type=int, dest='epoch', help='用于训练 CViT 模型的 epoch 数.')
parser.add_option("-v", "--version", dest='version', help='Version 0.1.')
parser.add_option("-s", "--cession", type="string", dest='session', help='Training session. Use g for GPU, t for TPU.')
parser.add_option("-d", "--dir", dest='dir', help='Training data path.')
parser.add_option("-b", "--batch", type=int, dest='batch', help='Batch size.')
parser.add_option("-l", "--rate", type=float, dest='rate', help='Learning rate.')
parser.add_option("-w", "--decay", type=float, dest='decay', help='Weight decay.')

(options, args) = parser.parse_args()

if options.session:
    cession = options.session
if options.dir == None:
    print(parser.usage)
    # exit(0) # 如果训练目录为None则退出程序
else:
    dir_path = options.dir
if options.batch:
    batch_size = int(options.batch)
if options.epoch:
    epoch = int(options.epoch)
if options.rate:
    lr = float(options.rate)
if options.decay:
    weight_decay = float(options.decay)

batch_size, dataloaders, dataset_sizes = session(cession, dir_path, batch_size)

# CViT model definition
model = resnet34(set_device = device,num_classes=2,
                include_top=False,
                include_top_kan=True)
model.to(device)
# 加载预训练模型进行后续的训练
checkpoint = torch.load('/home/AIBike_ViG/interns/chenao/rknnkit/test/CViT-main/ResKan/result/KanRes34_deepfake_last2.pth', map_location=torch.device('cuda'))

model.load_state_dict(checkpoint)


# 设置权重衰减（weight decay）为weight_decay，用于控制参数更新时的正则化项
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)
num_epochs = epoch

scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)


def train_gpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss, min_val_acc):
    min_acc = min_val_acc
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    min_loss = min_val_loss

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []

    # with open('weight/cvit_deepfake_detection_ep_50.pkl', 'rb') as f:
    #    train_loss, train_accu, val_loss, val_accu = pickle.load(f)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            phase_idx = 0
            # 循环访问数据。
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # break
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()  # GPU || CPU

                if phase_idx % 20 == 0:
                    print(phase, ' loss:', phase_idx, ':', loss.item())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, phase_idx * batch_size, dataset_sizes[phase], \
                        100. * phase_idx * batch_size / dataset_sizes[phase], loss.item()))
                phase_idx += 1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = epoch_acc.cpu().numpy()
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_accu.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_accu.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > min_acc:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_loss, epoch_loss))
                min_acc = epoch_acc
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    save_path = "./ResKan/result"
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    # 保存最后epoch训练出的模型
    torch.save(model.state_dict(), "./ResKan/result/KanRes34_deepfake_last3.pth")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # epoch全部训练完之后保存最优模型
    model.load_state_dict(best_model_wts)

    with open('./ResKan/result/KanRes34_deepfake_v3.pkl', 'wb') as f:
        pickle.dump([train_loss, train_accu, val_loss, val_accu], f)

    list = pd.DataFrame({"train_loss":train_loss, "train_accu":train_accu, "val_loss":val_loss, "val_acc":val_accu})
    list.to_csv('./ResKan/result/KanRes34_deepfake_v3.csv', index=False)

    state = {'epoch': num_epochs + 1,
             'state_dict': model.state_dict(),
             'min_loss': epoch_loss}
    torch.save(state, './ResKan/result/KanRes34_deepfake_v3.pth')
    test(model)
    return train_loss, train_accu, val_loss, val_accu, min_loss


def test(model):
    model.eval()

    Sum = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs).to(device).float()

        _, prediction = torch.max(output, 1)

        pred_label = labels[prediction]
        pred_label = pred_label.detach().cpu().numpy()
        main_label = labels.detach().cpu().numpy()
        bool_list = list(map(lambda x, y: x == y, pred_label, main_label))
        Sum += sum(np.array(bool_list) * 1)

    print('Prediction: ', (Sum / dataset_sizes['test']) * 100, '%')


# GPU进行运算
train_gpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss, min_val_acc)  # Train using GPU.