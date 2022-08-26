import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms

from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import os
import random
import cv2
import timm
import argparse
import datetime

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=1, help='')
parser.add_argument('--n_epochs', type=int, default=50, help='')
parser.add_argument('--data_folder', type=str, default='', help='')
parser.add_argument('--optimizer_class', type=str, default='Adam', help='')
parser.add_argument('--output_dir_name', type=str, default='FineTunedModels', help='')
parser.add_argument('--imagemodel', type=str, default='vgg19', help='')
parser.add_argument('--lr_value', type=float, default=1e-4, help='')
parser.add_argument('--batch_size_value', type=int, default=16, help='')
parser.add_argument('--layer_depth_vgg', type=int, default=8, help='')
args = parser.parse_args()

training_switch = True
current_dir = os.getcwd()

layer_depth = args.layer_depth_vgg

device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
n_gpus = torch.cuda.device_count()

image_model_name_of_timm = args.imagemodel
optimizer_class = args.optimizer_class
lr = args.lr_value

batch_size = args.batch_size_value
beta1 = 0.9
beta2 = 0.999
epochs = args.n_epochs

# gpu_ids = list(range(n_gpus))
gpu_ids = [args.gpu_id]

our_model = timm.create_model(image_model_name_of_timm, pretrained=True)

transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
if args.imagemodel == 'vgg19' or args.imagemodel == 'resnet50':
    pass
elif args.imagemodel == 'inception_v3':
    transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor()])
else:
    print('Specify pre-trained image model which you use for this fine-tuning.')
    sys.exit()

train_dataset_path = os.path.join(current_dir, args.data_folder + '/' + 'train')
test_dataset_path = os.path.join(current_dir, args.data_folder + '/' + 'test')
train_data = ImageFolder(train_dataset_path, transform)
test_data = ImageFolder(test_dataset_path, transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

all_labels = []
for _sample in train_loader:
    data, label = _sample
    all_labels.append(label.tolist())
max(sum(all_labels, []))

if args.imagemodel == 'vgg19':
    num_features = our_model.head.fc.in_features
    our_model.head.fc = nn.Linear(num_features, max(sum(all_labels, []))+1)
    if training_switch:
        for layer in list(our_model.children())[0][:-layer_depth]:
            layer.trainable = False
        for layer in list(our_model.children())[0][-layer_depth:]:
            layer.trainable = True
    else:
        pass
elif args.imagemodel == 'resnet50':
    num_features = our_model.fc.in_features
    our_model.fc = nn.Linear(num_features, max(sum(all_labels, []))+1)
    if training_switch:
        our_model.conv1.trainable = False
        our_model.bn1.trainable = False
        our_model.act1.trainable = False
        our_model.maxpool.trainable = False
        our_model.layer1.trainable = False
        our_model.layer2.trainable = False
        our_model.layer3.trainable = False
        our_model.layer4.trainable = False
        our_model.layer4[1].trainable = True
        our_model.layer4[2].trainable = True
elif args.imagemodel == 'inception_v3':
    num_features = our_model.fc.in_features
    our_model.fc = nn.Linear(num_features, max(sum(all_labels, []))+1)
    if training_switch:
        our_model.Conv2d_1a_3x3.trainable = False
        our_model.Conv2d_2a_3x3.trainable = False
        our_model.Conv2d_2b_3x3.trainable = False
        our_model.Pool1.trainable = False
        our_model.Conv2d_3b_1x1.trainable = False
        our_model.Conv2d_4a_3x3.trainable = False
        our_model.Pool2.trainable = False
        our_model.Mixed_5b.trainable = False
        our_model.Mixed_5c.trainable = False
        our_model.Mixed_5d.trainable = False
        our_model.Mixed_6a.trainable = False
        our_model.Mixed_6b.trainable = False
        our_model.Mixed_6c.trainable = False
        our_model.Mixed_6d.trainable = False
        our_model.Mixed_6e.trainable = False
        our_model.Mixed_7a.trainable = False
        our_model.Mixed_7b.trainable = False
        our_model.Mixed_7c.trainable = True

model = our_model.to(device)
model = torch.nn.DataParallel(model, device_ids=gpu_ids)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-08, weight_decay=0)
if optimizer_class == 'Adam':
    pass
elif optimizer_class == 'Adagrad':
    optimizer = optim.Adagrad(
        model.parameters(),
        lr=lr,
        lr_decay=0,
        weight_decay=1e-6,
        initial_accumulator_value=0,
        eps=1e-10
    )
elif optimizer_class == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)

schedular = ExponentialLR(optimizer, gamma=0.95)

epoch_loss_values = []
correct_values = []
total_values = []
test_accuracy = 0

for epoch in range(epochs):
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    model.train()
    for i, samples in enumerate(train_loader):
        data, labels = samples
        
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        
        predicted = outputs.max(1,  keepdim=True)[1]

        correct += predicted.eq(labels.view_as(predicted)).sum().item()
        total += labels.size(0)
        
        loss.backward()
        optimizer.step()

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1, keepdim=True)
            test_correct += predicted.eq(labels.view_as(predicted)).sum().item()
            test_total += labels.size(0)

    test_accuracy = test_correct / test_total
    
    epoch_loss_values_ = epoch_loss / len(train_loader)
    correct_values_ = correct / total
    total_values_ = test_accuracy
    
    epoch_loss_values.append(epoch_loss_values_)
    correct_values.append(correct_values_)
    total_values.append(total_values_)
    
    print("{}epoch: Loss {}, Accuracy {}, Test Accuracy {}".format(
        (epoch+1),
        epoch_loss / len(train_loader),
        correct / total,
        test_accuracy
    ))

print("------ Finish Training ------")

_base_dir = os.path.join(current_dir, args.output_dir_name)

if not os.path.exists(_base_dir):
    os.makedirs(_base_dir)

dt_now = datetime.datetime.now()

model_path = _base_dir + '/'\
 + '_baseImageModel_is_' + str(image_model_name_of_timm)\
 + '_finalTestAccuracy_is_' + str(test_accuracy)\
 + '_withDataSet_' + str(train_dataset_path.split('/')[-2].split('datasets_')[-1])\
 + '_Seed_' + str(train_dataset_path.split('seed__')[-1].split('__TrashTHRatio')[0])\
 + '_AfterPreTrained_epochs' + '_' + str(epochs) + '_lr_' + str(lr)\
 + '_BatchSize_' + str(batch_size)\
 + '_optim_' + str(optimizer_class)\
 + '_datetime_' + str(dt_now).split(' ')[-2] + '_'\
 + str(dt_now).split(' ')[-1].replace(':', '_').replace('.', '_')\
 + '.pt'
torch.save(model.to('cpu').state_dict(), model_path)

model_result_path = _base_dir + '/'\
 + '_baseImageModel_is_' + str(image_model_name_of_timm)\
 + '_finalTestAccuracy_is_' + str(test_accuracy)\
 + '_withDataSet_' + str(train_dataset_path.split('/')[-2].split('datasets_')[-1])\
 + '_Seed_' + str(train_dataset_path.split('seed__')[-1].split('__TrashTHRatio')[0])\
 + '_AfterPreTrained_epochs' + '_' + str(epochs) + '_lr_' + str(lr)\
 + '_BatchSize_' + str(batch_size)\
 + '_optim_' + str(optimizer_class)\
 + '_datetime_' + str(dt_now).split(' ')[-2] + '_'\
 + str(dt_now).split(' ')[-1].replace(':', '_').replace('.', '_')\
 + '__'

pd.DataFrame(epoch_loss_values).to_csv(model_result_path + 'loss.txt', index=None)
pd.DataFrame(correct_values).to_csv(model_result_path + 'train_accuracy.txt', index=None)
pd.DataFrame(total_values).to_csv(model_result_path + 'test_accuracy.txt', index=None)
