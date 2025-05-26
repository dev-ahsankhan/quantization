import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import timeit

from adapt.examples.models.resnet import resnet18, resnet34, resnet50
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib

# Configuration
threads = 40
torch.set_num_threads(threads)
axx_mult = 'mul8s_acc'

# Load pretrained model using approximate multiplier
model = resnet50(pretrained=True, axx_mult=axx_mult)

# Prepare CIFAR-10 dataset
def val_dataloader(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    dataset = CIFAR10(root="datasets/cifar10_data", train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=128, num_workers=0, drop_last=True, pin_memory=False)

transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])
dataset = CIFAR10(root="datasets/cifar10_data", train=True, download=True, transform=transform)

# Prepare training subset for calibration
evens = list(range(0, len(dataset), 10))
trainset_1 = torch.utils.data.Subset(dataset, evens)
data_t = DataLoader(trainset_1, batch_size=128, shuffle=False, num_workers=0)

# Run model calibration for quantization
def collect_stats(model, data_loader, num_batches):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (images, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(images)
        if i >= num_batches:
            break

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.load_calib_amax(**kwargs)
            print(f"{name:40}: {module}")
    model.cpu()

# Calibration: adjust quantization parameters
with torch.no_grad():
    collect_stats(model, data_t, num_batches=10)
    amax = compute_amax(model, method="percentile", percentile=99.99)

# Define criterion and optimizer for training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model with QAT setup
train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
for epoch in range(1):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Train Loss: {running_loss / len(train_loader):.4f}")

# Evaluate model after QAT
model.eval()
correct = 0
total = 0
val_loader = val_dataloader()
start_time = timeit.default_timer()
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
elapsed_time = timeit.default_timer() - start_time
print(f"Evaluation Time: {elapsed_time:.4f} seconds")
print(f'Accuracy: {(100 * correct / total):.2f} %')