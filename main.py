import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Import our custom AdaPT layers with Brevitas quantization
from adapt.adapt.approx_layers.axx_layers import AdaPT_Conv2d_Brevitas, AdaPT_Linear_Brevitas

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST dataset
train_data = datasets.MNIST(root='./cnn_data', train=True, download=True, 
                          transform=transforms.ToTensor())
test_data = datasets.MNIST(root='./cnn_data', train=False, download=True,
                         transform=transforms.ToTensor())

# Data loader
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Define LeNet-5 with our custom AdaPT layers
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = AdaPT_Conv2d_Brevitas(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = AdaPT_Conv2d_Brevitas(6, 16, kernel_size=5, stride=1)
        self.fc1 = AdaPT_Linear_Brevitas(16*5*5, 120)
        self.fc2 = AdaPT_Linear_Brevitas(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Final layer remains unquantized

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Create model, criterion, and optimizer
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(epochs=5):
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = train_correct = 0

        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(y_pred.data, 1)
            train_correct += (predicted == y_train).sum().item()
            train_loss += loss.item()

        # Calculate epoch metrics
        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / len(train_data)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

        # Test evaluation
        model.eval()
        test_loss = test_correct = 0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                y_pred = model(x_test)
                test_loss += criterion(y_pred, y_test).item()
                _, predicted = torch.max(y_pred.data, 1)
                test_correct += (predicted == y_test).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100. * test_correct / len(test_data)
        print(f'Epoch {epoch+1}: Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

    print(f'Training completed in {(time.time()-start_time)/60:.2f} minutes')

# Start training
train_model()