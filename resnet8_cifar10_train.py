import sys
sys.path.insert(1, 'examples/')

import os
import zipfile
import torch

import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn as nn

from models.resnet8 import resnet8
from models.data import get_test_dataloader
from models.data import get_training_dataloader

threads = 32
torch.set_num_threads(threads)

axx_mult = 'mul8s_acc'

model = resnet8(axx_mult)
model.load_state_dict(torch.load('examples/models/state_dicts/resnet8.pth', map_location="cpu").state_dict())
model.eval() # for evaluation

val_loader = get_test_dataloader(
        batch_size=32,
        num_workers=0,
        shuffle=False
    )

transform_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
        ]
)
cifar10_training = CIFAR10(
        root="cifar-10-torch",
        train=True,
        download=True,
        transform=transform_train,
)
evens = list(range(0, len(cifar10_training), 10))
trainset_1 = torch.utils.data.Subset(cifar10_training, evens)

train_loader = DataLoader(
        trainset_1,
        shuffle=False,
        num_workers=0,
        batch_size=128,
    )

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib

def collect_stats(model, data_loader, num_batches):
     """Feed data to the network and collect statistic"""

     # Enable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.disable_quant()
                 module.enable_calib()
             else:
                 module.disable()

     for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
         model(image.cpu())
         if i >= num_batches:
             break

     # Disable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.enable_quant()
                 module.disable_calib()
             else:
                 module.enable()

def compute_amax(model, **kwargs):
 # Load calib result
 for name, module in model.named_modules():
     if isinstance(module, quant_nn.TensorQuantizer):
         if module._calibrator is not None:
             if isinstance(module._calibrator, calib.MaxCalibrator):
                 module.load_calib_amax()
             else:
                 module.load_calib_amax(**kwargs)
         print(F"{name:40}: {module}")
 model.cpu()

# It is a bit slow since we collect histograms on CPU
with torch.no_grad():
    stats = collect_stats(model, train_loader, num_batches=2)
    amax = compute_amax(model, method="percentile", percentile=99.99)
    
    # optional - test different calibration methods
    #amax = compute_amax(model, method="mse")
    #amax = compute_amax(model, method="entropy")

##############################
## Run model for evaluation
##############################

import timeit
correct = 0
total = 0

model.eval()
start_time = timeit.default_timer()
with torch.no_grad():
    for iteraction, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
        images, labels = images.to("cpu"), labels.to("cpu")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(timeit.default_timer() - start_time)
print('Accuracy of the network on the 10000 test images: %.4f %%' % (
    100 * correct / total))

######################################
## Run Approximation-Aware retraining
######################################
from adapt.references.classification.train import evaluate, train_one_epoch, load_data

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.000025)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# finetune the model for one epoch based on data_t subset 
train_one_epoch(model, criterion, optimizer, train_loader, "cpu", 0, 1)


######################################
## Rerun mode evaluation
######################################
correct = 0
total = 0

model.eval()
start_time = timeit.default_timer()
with torch.no_grad():
    for iteraction, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
        images, labels = images.to("cpu"), labels.to("cpu")
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(timeit.default_timer() - start_time)
print('Accuracy of the network on the 10000 test images: %.4f %%' % (
    100 * correct / total))

torch.save(model.state_dict(), 'resnet8_adaptAAT.pt')