# modules

import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import timm
import argparse
import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from PIL import Image
import sys
import shutil
from collections import OrderedDict
import random

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

# input parameters

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=1, help='')
parser.add_argument('--model_class', type=str, default='vgg19', help='')
parser.add_argument('--output_dir_name', type=str, default='', help='')
parser.add_argument('--SideCuneiformDataset_dir', type=str, default='', help='')
parser.add_argument('--FrontBottomCuneiformDataset_dir', type=str, default='', help='')
parser.add_argument('--imagemodels_folder_name', type=str, default='', help='')
parser.add_argument('--batch_size_value', type=int, default=16, help='')
args = parser.parse_args()

# fixing seed

torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

# preparing output folder

current_dir: str = os.getcwd()
if not os.path.isdir(os.path.join(current_dir, args.output_dir_name)):
    os.mkdir(os.path.join(current_dir, args.output_dir_name))
else:
    print('Specified directory has already exist.')
    sys.exit()

# preparing dataset folders

folder_candidate_list = os.listdir(current_dir)

folder_list = []
for idx in folder_candidate_list:
    if os.path.isdir(os.path.join(current_dir, idx)):
        if idx.startswith('__TrainTestRatio')\
                and\
                (
                        idx.split('__')[-1] == 'v01'
                        or idx.split('__')[-1] == 'v02'
                        or idx.split('__')[-1] == 'v03'
                        or idx.split('__')[-1] == 'v04'
                        or idx.split('__')[-1] == 'v001'
                        or idx.split('__')[-1] == 'v002'
                        or idx.split('__')[-1] == 'v003'
                        or idx.split('__')[-1] == 'v004'
                ):
            folder_list.append(os.path.join(current_dir, idx))

# preparing a list of fine-tuned models

imagemodelfile_list_ = glob.glob(os.path.join(os.path.join(current_dir, args.imagemodels_folder_name), "**.pt"), recursive=True)

imagemodelfile_list = []
for idx in imagemodelfile_list_:
    if idx.split('/')[-1].startswith('_baseImageModel_is_'):
        imagemodelfile_list.append(idx)            

# setting GPU

n_gpus = torch.cuda.device_count()
# gpu_ids = list(range(n_gpus))
gpu_ids = [args.gpu_id]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# generating cases

def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict

model_lists = ['vgg19', 'resnet50', 'inception']

for model_ in model_lists:
    zeros_class4 = np.zeros((4, 4))
    zeros_class8 = np.zeros((8, 8))
    for idx_imagemodelfile in range(len(imagemodelfile_list)):
        for idx_folder in range(len(folder_list)):
            if imagemodelfile_list[idx_imagemodelfile].split('_is_')[1].split('_')[0] == model_:
                if folder_list[idx_folder].split('datasets_')[-1] \
                        == imagemodelfile_list[idx_imagemodelfile].split('DataSet_')[-1].split('_Seed')[0]:
                    folder_name = folder_list[idx_folder]
                    model_path = imagemodelfile_list[idx_imagemodelfile]
                    n_classes = len(os.listdir(os.path.join(folder_name, 'train')))
                    test_dataset_path = os.path.join(folder_name, 'test')

                    batch_size = args.batch_size_value
                    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

                    if model_ == 'vgg19' or model_ == 'resnet50':
                        pass
                    elif model_ == 'inception':
                        transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor()])
                    else:
                        pass

                    test_data = ImageFolder(test_dataset_path, transform)
                    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

                    model = timm.create_model('vgg19', pretrained=True)

                    if model_ == 'vgg19':
                        num_features = model.head.fc.in_features
                        model.head.fc = nn.Linear(num_features, n_classes)
                    elif model_ == 'resnet50':
                        model = timm.create_model('resnet50', pretrained=True)
                        num_features = model.fc.in_features
                        model.fc = nn.Linear(num_features, n_classes)
                    elif model_ == 'inception':
                        model = timm.create_model('inception_v3', pretrained=True)
                        num_features = model.fc.in_features
                        model.fc = nn.Linear(num_features, n_classes)
                    else:
                        pass

                    model.load_state_dict(fix_key(torch.load(model_path)))

                    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
                    model = model.to(device)

                    # model fixing
                    model.eval()

                    predicted_tensor = torch.empty(1)
                    label_tensor = torch.empty(1)

                    probs_for_labels = []

                    with torch.no_grad():
                        for i, (data, labels) in enumerate(test_loader):
                            data = data.to(device)
                            labels = labels.to(device)
                            label_tensor = label_tensor.to(device)

                            label_tensor = torch.cat([label_tensor, labels], dim=0)

                            predicted_tensor = predicted_tensor.to(device)

                            outputs = model(data)

                            m = nn.Softmax(dim=1)
                            outputs_prob = m(outputs)
                            AA = outputs_prob.clone()
                            AA = AA.view(outputs.size(0), -1)
                            AA -= AA.min(1, keepdim=True)[0] 
                            AA /= AA.max(1, keepdim=True)[0] 
                            AA = AA.view(outputs.shape[0], -1)
                            probs_for_labels += np.array(AA.cpu()).tolist()

                            _, predicted = outputs.max(1, keepdim=True)
                            predicted_tensor = torch.cat([predicted_tensor, predicted.squeeze()], dim=0)

                    y_test = label_tensor.tolist()[1:]
                    fin_test_pred = predicted_tensor.squeeze().tolist()[1:]
                    y_unique = list(set(y_test))
                    cm = confusion_matrix(y_test, fin_test_pred, labels=y_unique)

                    if n_classes == 4:
                        zeros_class4 += cm
                    elif n_classes == 8:
                        zeros_class8 += cm

                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm,
                        display_labels=['C1', 'C2', 'C3', 'C4']
                    )
                    if n_classes == 4:
                        pass
                    elif n_classes == 8:
                        disp = ConfusionMatrixDisplay(
                            confusion_matrix=cm,
                            display_labels=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
                        )

                    if n_classes == 4:
                        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
                        plt.rcParams['font.size'] = 14
                        disp.plot(ax=ax, cmap='Blues')
                        plt.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
                        plt.tick_params(bottom=True, left=True, right=False, top=False)
                        plt.savefig(
                            os.path.join(
                                os.path.join(current_dir, args.output_dir_name),
                                model_
                                + '_DataSetType_'
                                + str(imagemodelfile_list[idx_imagemodelfile].split('SetType__')[-1].split('_Seed')[0])
                                + '_Seed_'
                                + str(imagemodelfile_list[idx_imagemodelfile].split('_Seed_')[-1].split('_After')[0])
                                + '_TestAccuracy_'
                                + str(imagemodelfile_list[idx_imagemodelfile].split('racy_is_')[-1].split('_withD')[0])
                                + '_.jpg'
                            ),
                            dpi=300
                        )
                        plt.clf()
                        plt.close()
                    elif n_classes == 8:
                        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
                        plt.rcParams['font.size'] = 14
                        disp.plot(ax=ax, cmap='Blues')
                        plt.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
                        plt.tick_params(bottom=True, left=True, right=False, top=False)
                        plt.savefig(
                            os.path.join(
                                os.path.join(current_dir, args.output_dir_name),
                                model_
                                + '_DataSetType_'
                                + str(imagemodelfile_list[idx_imagemodelfile].split('SetType__')[-1].split('_Seed')[0])
                                + '_Seed_'
                                + str(imagemodelfile_list[idx_imagemodelfile].split('_Seed_')[-1].split('_After')[0])
                                + '_TestAccuracy_'
                                + str(imagemodelfile_list[idx_imagemodelfile].split('racy_is_')[-1].split('_withD')[0])
                                + '_.jpg'
                            ),
                            dpi=300
                        )
                        plt.clf()
                        plt.close()
                    else:
                        pass

                    # testing side cuneiform sentences

                    __type = 'side'

                    ext_folder_name = os.path.join(current_dir, args.SideCuneiformDataset_dir)

                    ext_data = ImageFolder(ext_folder_name, transform)
                    ext_loader = DataLoader(ext_data, batch_size=batch_size, shuffle=True, drop_last=True)

                    predicted_tensor_ext = torch.empty(1)
                    label_tensor_ext = torch.empty(1)

                    probs_for_labels_ext = []

                    with torch.no_grad():
                        for i, (data, labels) in enumerate(ext_loader):
                            data = data.to(device)
                            labels = labels.to(device)
                            label_tensor_ext = label_tensor_ext.to(device)

                            label_tensor_ext = torch.cat([label_tensor_ext, labels], dim=0)

                            predicted_tensor_ext = predicted_tensor_ext.to(device)

                            outputs_ext = model(data)

                            m = nn.Softmax(dim=1)
                            outputs_prob_ext = m(outputs_ext)
                            AA_ext = outputs_prob_ext.clone()
                            AA_ext = AA_ext.view(outputs_ext.size(0), -1)
                            AA_ext -= AA_ext.min(1, keepdim=True)[0] 
                            AA_ext /= AA_ext.max(1, keepdim=True)[0] 
                            AA_ext = AA_ext.view(outputs_ext.shape[0], -1)
                            probs_for_labels_ext += np.array(AA_ext.cpu()).tolist()

                            _, predicted_ext = outputs_ext.max(1, keepdim=True)
                            predicted_tensor_ext = torch.cat([predicted_tensor_ext, predicted_ext.squeeze()], dim=0)

                    plt.rcParams['font.size'] = 14
                    plt.rcParams["figure.figsize"] = (10, int(len(probs_for_labels_ext)/4))
                    g = sns.heatmap(probs_for_labels_ext, linewidths=.5, linecolor="Blue", cmap='Blues', cbar=False)
                    if n_classes == 8:
                        g.set_xticklabels(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
                    elif n_classes == 4:
                        g.set_xticklabels(['C1', 'C2', 'C3', 'C4'])
                    else:
                        pass
                    plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
                    plt.tick_params(bottom=False, left=False, right=False, top=False)
                    plt.savefig(
                        os.path.join(
                            os.path.join(current_dir, args.output_dir_name),
                            model_
                            + '_DataSetType_'
                            + str(imagemodelfile_list[idx_imagemodelfile].split('SetType__')[-1].split('_Seed')[0])
                            + '_Seed_'
                            + str(imagemodelfile_list[idx_imagemodelfile].split('_Seed_')[-1].split('_After')[0])
                            + '_TestAccuracy_'
                            + str(imagemodelfile_list[idx_imagemodelfile].split('racy_is_')[-1].split('_withD')[0])
                            + '_'
                            + str(__type)
                            + '.jpg'
                        ),
                        dpi=300
                    )
                    plt.clf()
                    plt.close()

                    __type = 'oldclass4'

                    ext_folder_name = os.path.join(current_dir, args.FrontBottomCuneiformDataset_dir)

                    ext_data = ImageFolder(ext_folder_name, transform)
                    ext_loader = DataLoader(ext_data, batch_size=batch_size, shuffle=True, drop_last=True)

                    predicted_tensor_ext = torch.empty(1)
                    label_tensor_ext = torch.empty(1)

                    probs_for_labels_ext = []

                    with torch.no_grad():
                        for i, (data, labels) in enumerate(ext_loader):
                            data = data.to(device)
                            labels = labels.to(device)
                            label_tensor_ext = label_tensor_ext.to(device)

                            label_tensor_ext = torch.cat([label_tensor_ext, labels], dim=0)

                            predicted_tensor_ext = predicted_tensor_ext.to(device)

                            outputs_ext = model(data)

                            m = nn.Softmax(dim=1)
                            outputs_prob_ext = m(outputs_ext)
                            AA_ext = outputs_prob_ext.clone()
                            AA_ext = AA_ext.view(outputs_ext.size(0), -1)
                            AA_ext -= AA_ext.min(1, keepdim=True)[0] 
                            AA_ext /= AA_ext.max(1, keepdim=True)[0] 
                            AA_ext = AA_ext.view(outputs_ext.shape[0], -1)
                            probs_for_labels_ext += np.array(AA_ext.cpu()).tolist()

                            _, predicted_ext = outputs_ext.max(1, keepdim=True)
                            predicted_tensor_ext = torch.cat([predicted_tensor_ext, predicted_ext.squeeze()], dim=0)

                    plt.rcParams['font.size'] = 14
                    plt.rcParams["figure.figsize"] = (10, int(len(probs_for_labels_ext)/4))
                    g = sns.heatmap(probs_for_labels_ext, linewidths=.5, linecolor="Blue", cmap='Blues', cbar=False)
                    if n_classes == 8:
                        g.set_xticklabels(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
                    elif n_classes == 4:
                        g.set_xticklabels(['C1', 'C2', 'C3', 'C4'])
                    else:
                        pass
                    plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
                    plt.tick_params(bottom=False, left=False, right=False, top=False)
                    plt.savefig(
                        os.path.join(
                            os.path.join(current_dir, args.output_dir_name),
                            model_
                            + '_DataSetType_'
                            + str(imagemodelfile_list[idx_imagemodelfile].split('SetType__')[-1].split('_Seed')[0])
                            + '_Seed_'
                            + str(imagemodelfile_list[idx_imagemodelfile].split('_Seed_')[-1].split('_After')[0])
                            + '_TestAccuracy_'
                            + str(imagemodelfile_list[idx_imagemodelfile].split('racy_is_')[-1].split('_withD')[0])
                            + '_'
                            + str(__type)
                            + '.jpg'
                        ),
                        dpi=300
                    )
                    plt.clf()
                    plt.close()

                else:
                    pass
            else:
                pass

    # generating overall cases

    disp = ConfusionMatrixDisplay(
        confusion_matrix=zeros_class4.astype(int),
        display_labels=['C1', 'C2', 'C3', 'C4']
    )
    _, ax = plt.subplots(figsize=(5, 5), dpi=300)
    plt.rcParams['font.size'] = 12
    disp.plot(ax=ax, cmap='Blues')
    # ax.tick_params(labelbottom=True,labelleft=False,labelright=False,labeltop=False)
    # ax.tick_params(bottom=False,left=False,right=False,top=False)
    # plt.title('Overall Results on VGG19, 4 Classes')
    plt.savefig(
        os.path.join(
            os.path.join(current_dir, args.output_dir_name),
            model_ + '_overallresult_4classes_.jpg'
        ),
        dpi=300
    )
    plt.clf()
    plt.close()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=zeros_class8.astype(int),
        display_labels=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    )
    _, ax = plt.subplots(figsize=(5, 5), dpi=300)
    plt.rcParams['font.size'] = 9
    disp.plot(ax=ax, cmap='Blues')
    # plt.title('Overall Results on VGG19, 8 Classes')
    plt.savefig(
        os.path.join(
            os.path.join(current_dir, args.output_dir_name),
            model_ + '_overallresult_8classes_.jpg'
        ),
        dpi=300
    )
    plt.clf()
    plt.close()
    # plt.show()
