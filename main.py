#main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
import time

######################## ADAPT ########################
from adapt.adapt.approx_layers import axx_layers as approxNN

# Set flag for use of AdaPT custom layers or vanilla PyTorch 
use_adapt = True

# Set axx mult. default = accurate 
axx_mult_global = 'mul8s_acc'
#######################################################

# Device configuration: Use GPU if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform 
transform = transforms.Compose([ 
    transforms.ToTensor(), 
])

# Data 
train_data = datasets.MNIST(root='./cnn_data', train=True, download=True, transform=transform) 
test_data = datasets.MNIST(root='./cnn_data', train=False, download=True, transform=transform)

# Dataloaders with batch size set
train_loader = DataLoader(train_data, batch_size=64, shuffle=True) 
test_loader = DataLoader(test_data, batch_size=128, shuffle=False) 

# Model Class 
class ConvNetwork(nn.Module): 
    def __init__(self):
        super().__init__()
        # import pdb; pdb.set_trace()
        
        if use_adapt: 
            
            self.conv1 = approxNN.AdaPT_Conv2d(1, 6, 3) 
        else:
            self.conv1 = nn.Conv2d(1, 6, 3)

        self.conv2 = nn.Conv2d(6, 16, 3, 1) 
        self.fc1 = approxNN.AdaPT_Linear(5*5*16, 120) 
        self.fc2 = approxNN.AdaPT_Linear(120, 84) 
        self.fc3 = approxNN.AdaPT_Linear(84, 10)

    def forward(self, x): 
        x = F.relu(self.conv1(x)) 
        x = F.max_pool2d(x, 2, 2) 
        x = F.relu(self.conv2(x)) 
        x = F.max_pool2d(x, 2, 2) 
        x = x.view(-1, 16*5*5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x) 
        return F.log_softmax(x, dim=1)

def calculate_model_size(model): 
    return sum(p.numel() for p in model.parameters())

# Create an Instance of our Model on the selected device 
torch.manual_seed(41)
model = ConvNetwork().to(device)

# Print original model architecture and size
print("Original Model Architecture:")
print(model)
print("Model size: ", calculate_model_size(model))

# Loss Function and Optimizer 
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training setup 
start_time = time.time()

epochs = 5 
train_losses = [] 
test_losses = [] 
train_correct = [] 
test_correct = []

for epoch in range(epochs): 
    model.train()
    train_corr = 0 
    for batch_number, (x_train, y_train) in enumerate(train_loader, 1): 
        x_train, y_train = x_train.to(device), y_train.to(device) 
        y_pred = model(x_train) 
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1] 
        batch_corr = (predicted == y_train).sum() 
        train_corr += batch_corr

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()

        
       
        if batch_number % 600 == 0:
            print(f'Epoch: {epoch}  Batch: {batch_number}  Loss: {loss.item()}')

    train_losses.append(loss.item()) 
    train_correct.append(train_corr)

    model.eval() 
    test_corr = 0 
    with torch.no_grad():
        for x_test, y_test in test_loader: 
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_val = model(x_test) 
            predicted = torch.max(y_val.data, 1)[1] 
            test_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test) 
    test_losses.append(loss.item()) 
    test_correct.append(test_corr)

end_time = time.time()

# Print training time 
print(f'Training Took: {(end_time - start_time)/60:.2f} minutes!')

# Print accuracy results 
print(f"Accuracy: {test_correct[-1].item() / len(test_data)* 100:.2f}%")
