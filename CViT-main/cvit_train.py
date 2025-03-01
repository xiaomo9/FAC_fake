import torch
import sys, os
os.chdir(r'E:\Daima\face_fake\CViT-main')
import argparse
import numpy as np
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
from cvit import CViT
from loader import session
import optparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Default
cession='g' # GPU runtime 
epoch = 40
# dir_path = "/autodl-tmp/data/total_face"
dir_path = r'E:\Daima\dataset\FF++\FF++\ff_face_total_data\train_data\total'
batch_size = 32
lr=0.0001
min_val_loss=0.6
best_val_acc = 0.7
weight_decay=0.000001

parser = optparse.OptionParser("Train CViT model.")
parser.add_option("-e", "--epoch", type=int, dest='epoch', help='用于训练CViT模型轮数epoch')
parser.add_option("-v", "--version", dest='version', help='Version 0.1.')
parser.add_option("-s", "--cession", type="string",dest='session', help='Training session. Use g for GPU, t for TPU.')
parser.add_option("-d", "--dir", dest='dir', help='Training data path.')
parser.add_option("-b", "--batch", type=int, dest='batch', help='Batch size.')
parser.add_option("-l", "--rate",  type=float, dest='rate', help='Learning rate.')
parser.add_option("-w", "--decay", type=float, dest='decay', help='Weight decay.')

(options,args) = parser.parse_args()

if options.session:
    cession = options.session
if options.dir==None:
    print (parser.usage)
    # exit(0) # 如果训练目录为None则退出程序?
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

#CViT model definition
model = CViT(image_size=224, patch_size=7, num_classes=2, channels=512,
            dim=1024, depth=6, heads=8, mlp_dim=2048)
model.to(device)
# 加载预训练模型
# ==============================================================
checkpoint = torch.load(r'E:\Daima\face_fake\CViT-main\weight\cvit.pth', map_location=torch.device('cuda'))
model.load_state_dict(checkpoint,strict=False)
# ================================================================

# 设置权重衰减（weight decay）为weight_decay，用于控制参数更新时的正则化项
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)
num_epochs = epoch

# scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=3, verbose=True)
"""
optimer: 指的是网络的优化器
mode (str): 可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
factor: 学习率每次降低多少，new_lr = old_lr * factor
patience: 容忍网路的性能不提升的次数，高于这个次数就降低学习率
verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
cooldown： 减少lr后恢复正常操作之前要等待的时期数。 默认值：0。
min_lr: 学习率的下限
eps: 适用于lr的最小衰减。 如果新旧lr之间的差异小于eps，则忽略更新。 默认值：1e-8。
"""


def train_gpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    min_loss = min_val_loss
    best_acc = best_val_acc

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []

    #with open('weight/cvit_deepfake_detection_ep_50.pkl', 'rb') as f:
    #    train_loss, train_accu, val_loss, val_accu = pickle.load(f)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            phase_idx=0
            # 循环访问数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #break
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step() # GPU || CPU

                if phase_idx % 20 == 0:
                    print(phase,' loss:',phase_idx,':', loss.item())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, phase_idx * batch_size, dataset_sizes[phase], \
                        100. * phase_idx*batch_size / dataset_sizes[phase], loss.item()))
                phase_idx+=1                       

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = epoch_acc.cpu().numpy()

            # if phase == 'train':
            #     scheduler.step()
            if phase == 'validation':
                scheduler.step(epoch_loss)


            if phase == 'train':
                train_loss.append(epoch_loss)
                train_accu.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_accu.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                print('Validation acc increase ({:.6f} --> {:.6f}).  Saving model ...'.format(best_acc, epoch_acc))
                min_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {'epoch': num_epochs+1, 
                        'state_dict': best_model_wts,
                        'optimizer': optimizer.state_dict(),
                        'min_loss':epoch_loss}
                # 防止训练过程中意外关闭程序，一边训练一边保存模型?
                # torch.save(state, 'weight/cvit_deepfake_detection_dfdc_ff原.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
# 保存最后训练的模型
    # torch.save(model.state_dict(), 'weight/cvit_deepfake_detection_last_dfdc_ff原.pth')
# epoch全部训练完之后保存最优模型
    # load best model weights
    model.load_state_dict(best_model_wts)


    with open('weight/cvit_dfdc_FFtotal.pkl', 'wb') as f:
        pickle.dump([train_loss, train_accu, val_loss, val_accu], f)

    state = {'epoch': num_epochs+1, 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_loss':epoch_loss}
    torch.save(model.state_dict(), 'weight/cvit_dfdc_FFtotal.pth')
    test(model)
    return train_loss, train_accu, val_loss, val_accu, min_loss

def test(model):
    model.eval()

    Sum = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs).to(device).float()
        
        _,prediction = torch.max(output,1)
        
        # pred_label = labels[prediction]
        pred_label = prediction
        pred_label = pred_label.detach().cpu().numpy()
        main_label = labels.detach().cpu().numpy()
        bool_list  = list(map(lambda x, y: x == y, pred_label, main_label))
        Sum += sum(np.array(bool_list)*1)
        
    print('Prediction: ', (Sum/dataset_sizes['test'])*100,'%')
        
# GPU进行计算
if __name__ == '__main__':
    train_gpu(model, criterion, optimizer, scheduler, num_epochs, min_val_loss) #Train using GPU.