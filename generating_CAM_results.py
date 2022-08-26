# The main parts of the following codes are borrowed from: 
# https://github.com/jacobgil/pytorch-grad-cam

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
import timm
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from collections import OrderedDict
from typing import Callable, List, Tuple
import ttach as tta
import matplotlib.pyplot as plt
from kornia.filters.gaussian import gaussian_blur2d
from fastprogress.fastprogress import progress_bar
from PIL import Image
import os
import glob
import shutil
import argparse
import random
import sys

# input parameters

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=1, help='')
parser.add_argument('--output_dir_name', type=str, default='', help='Path to data.')
parser.add_argument('--VGG19_TargetFeaturesLayer_n', type=int, default='36', help='')
parser.add_argument('--imagemodels_folder_name', type=str, default='', help='Path to data.')
parser.add_argument('--batch_size_value', type=int, default=10, help='L2 regularisation')
args = parser.parse_args()


# fixing seed

torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


# preparing output dir

current_dir: str = os.getcwd()
output_path = os.path.join(current_dir, args.output_dir_name)
if not os.path.isdir(output_path):
    os.makedirs(output_path)
else:
    print('Specified directory has already exist.')
    sys.exit()
    
batch_size = args.batch_size_value


# utils

class ActivationsAndGradients:
    def __init__(self, model_, target_layers_, reshape_transform):
        self.model = model_
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers_:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result
    

def get_2d_projection(activation_batch):
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = activations.reshape(activations.shape[0], -1).transpose()
        reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)
    
def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict
    
def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1-image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam) 

def group_cluster(x, group=32, cluster_method='k_means'):
    # x : (torch tensor with shape [1, c, h, w])
    xs = x.detach().cpu()
    b, c, h, w = xs.shape
    xs = xs.reshape(b, c, -1).reshape(b*c, h*w)
    n_cluster = 0
    if cluster_method == 'k_means':
        n_cluster = KMeans(n_clusters=group, random_state=0).fit(xs)
    elif cluster_method == 'agglomerate':
        n_cluster = AgglomerativeClustering(n_clusters=group).fit(xs)
    else:
        assert NotImplementedError

    labels = n_cluster.labels_
    del xs
    return labels
    
def group_sum(x, n=32, cluster_method='k_means'):
    b, c, h, w = x.shape
    group_idx = group_cluster(x, group=n, cluster_method=cluster_method)
    init_masks = [torch.zeros(1, 1, h, w).to(x.device) for _ in range(n)]
    for i_ in range(c):
        idx_ = group_idx[i_]
        init_masks[idx_] += x[:, i_, :, :].unsqueeze(1)
    return init_masks
    
def preprocess_img(cv_img):
    # revert the channels from BGR to RGB
    img = cv_img.copy()[:, :, ::-1]
    # convert tor tensor
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1))))
    # Normalize
    transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    norm_img = transform_norm(img).unsqueeze(0)

    return img, norm_img
    
def show_cam(img, mask, title=None):
    mask = (mask - mask.min()).div(mask.max() - mask.min()).data
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().float()), cv2.COLORMAP_JET)  # [H, W, 3]
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    cam = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8) * heatmap
    if title is not None:
        vutils.save_image(cam, title)

    return cam

# URL: https://blog.shikoan.com/numpy-tile-images/
def make_grid(imgs, nrow, padding=0):
    assert imgs.ndim == 4 and nrow > 0
    batch, height, width, ch = imgs.shape
    n = nrow * (batch // nrow + np.sign(batch % nrow))
    ncol = n // nrow
    pad = np.zeros((n - batch, height, width, ch), imgs.dtype)
    x = np.concatenate([imgs, pad], axis=0)
    # border padding if required
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, padding), (0, padding), (0, 0)),
                   "constant", constant_values=(0, 0))
        height += padding
        width += padding
    x = x.reshape(ncol, nrow, height, width, ch)
    x = x.transpose([0, 2, 1, 3, 4])
    x = x.reshape(height * ncol, width * nrow, ch)
    if padding > 0:
        x = x[:(height * ncol - padding), :(width * nrow - padding), :]
    return x
    
# CAM algorithms

def aggregate_multi_layers(cam_per_target_layer: np.ndarray) -> np.ndarray:
    cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
    cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
    result = np.mean(cam_per_target_layer, axis=1)
    return scale_cam_image(result)

def get_target_width_height(input_tensor: torch.Tensor) -> Tuple[int, int]:
    width, height = input_tensor.size(-1), input_tensor.size(-2)
    return width, height

class BaseCAM(object):
    def __init__(self,
                 model_: torch.nn.Module,
                 target_layers_: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model_.eval()
        self.target_layers = target_layers_
        self.cuda = use_cuda
        if self.cuda:
            self.model = model_.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers_, reshape_transform)

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layer: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return aggregate_multi_layers(cam_per_layer)

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool
    ) -> np.ndarray:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for idx_ in range(len(self.target_layers)):
            target_layer = self.target_layers[idx_]
            layer_activations = None
            layer_grads = None
            if idx_ < len(activations_list):
                layer_activations = activations_list[idx_]
            if idx_ < len(grads_list):
                layer_grads = grads_list[idx_]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform_ in transforms:
            augmented_tensor = transform_.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform_.deaugment_mask(cam)

            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
            
# https://arxiv.org/abs/1610.02391
class GradCAM(BaseCAM):
    def __init__(
            self,
            model_,
            target_layers_,
            use_cuda=False,
            reshape_transform=None
    ):
        super(GradCAM, self).__init__(model_, target_layers_, use_cuda, reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))

# https://arxiv.org/abs/2008.00299
class EigenCAM(BaseCAM):
    def __init__(
            self,
            model_,
            target_layers_,
            use_cuda=False,
            reshape_transform=None
    ):
        super(EigenCAM, self).__init__(model_,
                                       target_layers_,
                                       use_cuda,
                                       reshape_transform,
                                       uses_gradients=False)

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:
        return get_2d_projection(activations)

# URL: https://arxiv.org/abs/1910.01279
class ScoreCAM(BaseCAM):
    def __init__(
            self,
            model_,
            target_layers_,
            use_cuda=False,
            reshape_transform=None):
        super(ScoreCAM, self).__init__(model_,
                                       target_layers_,
                                       use_cuda,
                                       reshape_transform=reshape_transform,
                                       uses_gradients=False)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            input_tensors = input_tensor[:, None,
                                         :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i_ in range(0, tensor.size(0), BATCH_SIZE):
                    batch = tensor[i_: i_ + BATCH_SIZE, :]
                    outputs = [target(o).cpu().item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights

# main process

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
gpu_ids = list(range(n_gpus))
# gpu_ids = [args.gpu_id]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_names = ['grad', 'eigen', 'score']

for idx_imagemodelfile in range(len(imagemodelfile_list)):
    for idx_folder in range(len(folder_list)):
        if imagemodelfile_list[idx_imagemodelfile].split('_is_')[1].split('_')[0] == 'vgg19':
            if folder_list[idx_folder].split('datasets_')[-1] == imagemodelfile_list[idx_imagemodelfile].split('DataSet_')[-1].split('_Seed')[0]:
                dataset_path = folder_list[idx_folder]
                model_path = imagemodelfile_list[idx_imagemodelfile]
                dataset_id = dataset_path.split('/')[-1]
                model_file = model_path.split('/')[-1]
                
                transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

                train_dataset_path = os.path.join(dataset_path, 'train')
                train_data = ImageFolder(train_dataset_path, transform)
                n_classes = len(train_data.class_to_idx)

                all_classes = []
                if n_classes == 4:
                    all_classes = ['class_01', 'class_02', 'class_03', 'class_04']
                elif n_classes == 8:
                    all_classes = ['class_01', 'class_02', 'class_03', 'class_04', 'class_05', 'class_06', 'class_07', 'class_08']
                else:
                    pass

                for _idx in all_classes:
    
                    _class_id = _idx
        
                    new_target_folder_path = os.path.join(current_dir, '_' + dataset_id + '_' + str(_class_id))
                    class_name = _class_id

                    if not os.path.exists(new_target_folder_path):
                        os.makedirs(new_target_folder_path)
                    else:
                        print('Specified directory has already exist.')
                        sys.exit()
                    if not os.path.exists(os.path.join(new_target_folder_path, 'test/' + _class_id)):
                        os.makedirs(os.path.join(new_target_folder_path, 'test/' + _class_id))
                    else:
                        print('Specified directory has already exist.')
                        sys.exit()
                    if not os.path.exists(os.path.join(new_target_folder_path, 'test/' + _class_id + '/data')):
                        os.makedirs(os.path.join(new_target_folder_path, 'test/' + _class_id + '/data'))

                    file_list_ = glob.glob(os.path.join(dataset_path, 'test/' + _class_id + "/**.jpg"), recursive=True)
                    for image_file_ in file_list_:
                        shutil.copy(image_file_, os.path.join(new_target_folder_path, 'test/' + _class_id + '/data'))

                    specific_category_dir = os.path.join(new_target_folder_path, 'test/' + _class_id)
                    specific_data = ImageFolder(specific_category_dir, transform)
                    specific_loader = DataLoader(specific_data, batch_size=batch_size, shuffle=True, drop_last=True)

                    model = timm.create_model('vgg19', pretrained=True)
                    num_features = model.head.fc.in_features
                    model.head.fc = nn.Linear(num_features, n_classes)
                    model.load_state_dict(fix_key(torch.load(model_path)))
                    model = model.to(device)
                    model.eval()

                    _target_layer = [model.features[args.VGG19_TargetFeaturesLayer_n]]

                    for i, data in enumerate(specific_loader, 0):
                        if i >= 1:  # If you need results for all i, turn off these following two lines.
                            break
                        for idx in model_names:
                            if idx == 'grad':
                                grad_result_list = []
                                for idy in range(data[0].shape[0]):
                                    target_layers = _target_layer
                                    CAM_model = GradCAM(
                                        model_=model,
                                        target_layers_=target_layers,
                                        use_cuda=True
                                    )
                                    input_data = data[0][idy].view(1, data[0][idy].shape[0], data[0][idy].shape[1], data[0][idy].shape[2])
                                    grayscale_cam = CAM_model(
                                        input_tensor=input_data,
                                        targets=[ClassifierOutputTarget(int(str(_class_id).split('_')[-1])-1)],
                                        aug_smooth=True,
                                        eigen_smooth=True
                                    )
                                    grayscale_cam = grayscale_cam[0, :]
                                    raw_image = data[0][idy].detach().numpy().copy()
                                    raw_image = raw_image.transpose((1, 2, 0))
                                    raw_image = ((raw_image * 0.5) + 0.5) * 255.0
                                    raw_image = raw_image.astype(np.uint8)
                                    raw_image_forGradCAMvis = np.float32(raw_image) / 255
                                    visualization = show_cam_on_image(raw_image_forGradCAMvis, grayscale_cam, use_rgb=True)
                                    grad_result_list.append(visualization)
                                grad_result_array = np.array(grad_result_list)
                                grad_stacked = make_grid(grad_result_array, 2, padding=2)
                                with Image.fromarray(grad_stacked) as grad_img:
                                    grad_img.save(
                                        os.path.join(
                                            output_path,
                                            'batchID_' + str(i).zfill(4)
                                            + '_dataset_'
                                            + str(imagemodelfile_list[idx_imagemodelfile].split('SetType__')[-1].split('_Seed')[0])
                                            + '_seed_'
                                            + str(imagemodelfile_list[idx_imagemodelfile].split('_Seed_')[-1].split('_After')[0])
                                            + '_CAMmodel_'
                                            + str(idx)
                                            + '_Class_'
                                            + str(_class_id)
                                            + '.jpg'
                                        ),
                                        dpi=(72, 72)
                                    )
                            elif idx == 'eigen':
                                eigen_result_list = []
                                for idy in range(data[0].shape[0]):
                                    target_layers = _target_layer
                                    CAM_model = EigenCAM(
                                        model_=model,
                                        target_layers_=target_layers,
                                        use_cuda=True
                                    )
                                    input_data = data[0][idy].view(1, data[0][idy].shape[0], data[0][idy].shape[1], data[0][idy].shape[2])
                                    grayscale_cam = CAM_model(
                                        input_tensor=input_data,
                                        targets=[ClassifierOutputTarget(int(str(_class_id).split('_')[-1])-1)],
                                        aug_smooth=True,
                                        eigen_smooth=True
                                    )
                                    grayscale_cam = grayscale_cam[0, :]
                                    raw_image = data[0][idy].detach().numpy().copy()
                                    raw_image = raw_image.transpose((1, 2, 0))
                                    raw_image = ((raw_image * 0.5) + 0.5) * 255.0
                                    raw_image = raw_image.astype(np.uint8)
                                    raw_image_forGradCAMvis = np.float32(raw_image) / 255
                                    visualization = show_cam_on_image(raw_image_forGradCAMvis, grayscale_cam, use_rgb=True)
                                    eigen_result_list.append(visualization)
                                eigen_result_array = np.array(eigen_result_list)
                                eigen_stacked = make_grid(eigen_result_array, 2, padding=2)
                                with Image.fromarray(eigen_stacked) as eigen_img:
                                    eigen_img.save(
                                        os.path.join(
                                            output_path,
                                            'batchID_'
                                            + str(i).zfill(4)
                                            + '_dataset_'
                                            + str(imagemodelfile_list[idx_imagemodelfile].split('SetType__')[-1].split('_Seed')[0])
                                            + '_seed_'
                                            + str(imagemodelfile_list[idx_imagemodelfile].split('_Seed_')[-1].split('_After')[0])
                                            + '_CAMmodel_'
                                            + str(idx)
                                            + '_Class_'
                                            + str(_class_id)
                                            + '.jpg'
                                        ),
                                        dpi=(72, 72)
                                    )
                            elif idx == 'score':
                                eigen_result_list = []
                                for idy in range(data[0].shape[0]):
                                    target_layers = _target_layer
                                    CAM_model = ScoreCAM(
                                        model_=model,
                                        target_layers_=target_layers,
                                        use_cuda=True
                                    )
                                    input_data = data[0][idy].view(1, data[0][idy].shape[0], data[0][idy].shape[1], data[0][idy].shape[2])
                                    grayscale_cam = CAM_model(
                                        input_tensor=input_data,
                                        targets=[ClassifierOutputTarget(int(str(_class_id).split('_')[-1])-1)],
                                        aug_smooth=True,
                                        eigen_smooth=True
                                    )
                                    grayscale_cam = grayscale_cam[0, :]
                                    raw_image = data[0][idy].detach().numpy().copy()
                                    raw_image = raw_image.transpose((1, 2, 0))
                                    raw_image = ((raw_image * 0.5) + 0.5) * 255.0
                                    raw_image = raw_image.astype(np.uint8)
                                    raw_image_forGradCAMvis = np.float32(raw_image) / 255
                                    visualization = show_cam_on_image(raw_image_forGradCAMvis, grayscale_cam, use_rgb=True)
                                    eigen_result_list.append(visualization)
                                eigen_result_array = np.array(eigen_result_list)
                                eigen_stacked = make_grid(eigen_result_array, 2, padding=2)
                                with Image.fromarray(eigen_stacked) as eigen_img:
                                    eigen_img.save(
                                        os.path.join(
                                            output_path,
                                            'batchID_'
                                            + str(i).zfill(4)
                                            + '_dataset_'
                                            + str(imagemodelfile_list[idx_imagemodelfile].split('SetType__')[-1].split('_Seed')[0])
                                            + '_seed_'
                                            + str(imagemodelfile_list[idx_imagemodelfile].split('_Seed_')[-1].split('_After')[0])
                                            + '_CAMmodel_'
                                            + str(idx)
                                            + '_Class_'
                                            + str(_class_id)
                                            + '.jpg'
                                        ),
                                        dpi=(72, 72)
                                    )

                    # cleaning
                    shutil.rmtree(new_target_folder_path)

    else:
        pass
