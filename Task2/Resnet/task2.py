# %%
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# %%
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%


# %%


# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # Convolutional layers
        self.convolutionLayer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride ,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
        
        self.convolutionLayer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        
        
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    # Other stuff
    def forward(self, x):
        residual = x
        output_layer1 = self.convolutionLayer1(x)
        output_layer2 = self.convolutionLayer2(output_layer1)
        
        # Skip Connection
        if self.downsample==True:
            residual = self.downsample(x)
        
            output_layer2 += residual
        
        
        # pass through the activation function
        output = self.relu(output_layer2)
        return output


# %%
class ResidualNeuralNetwork(nn.Module):
    
    def __init__(self, block, layers, num_classes=10):
        super(ResidualNeuralNetwork, self).__init__()
        self.inplanes = 64
        
        # Convolutional layers
        
        # in_channels = 3 as we have RGB images
        # First Convolution Layer with 64 Filters of size 7x7, stride=2 and padding=3
        self.convolutionLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Max Pooling layer of size 3x3, stride=2 and padding=1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Accordind to ResNet 34 Layer model we have 3,4,6,3 layers in each block with 64,128,256,512 filters respectively
        
        # 1st Block of 3 convolution layers of 64 filters of size 3x3
        self.layer1 = self.make_newLayer(block,64,layers[0],stride=1)
        
        # 2nd Block of 4 convolution layers of 128 filters of size 3x3
        self.layer2 = self.make_newLayer(block,128,layers[1],stride=2)
        
        # 3rd Block of 6 convolution layers of 256 filters of size 3x3
        self.layer3 = self.make_newLayer(block,256,layers[2],stride=2)
        
        # 4th Block of 3 convolution layers of 512 filters of size 3x3
        self.layer4 = self.make_newLayer(block,512,layers[3],stride=2)
        
        # Average Pooling Layer
        self.average_pool = nn.AvgPool2d(7, stride=1)
        
        # Fully Connected Layer consisting of 512 neurons
        self.FullyConnectedLayer = nn.Linear(512, num_classes)
    
    def make_newLayer(self, block, planes, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.inplanes != planes):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        #update the inplanes
        self.inplanes = planes
        
        #remaining feature maps(blocks)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    

    def forward(self, x):
        
        # Convolutional Layers
        output = self.convolutionLayer1(x)
        output = self.maxpool(output)
        
        # 4 Residual Blocks
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        
        # Average Pooling Layer
        output = self.average_pool(output)
        
        # Fully Connected Layer
        output = output.view(output.size(0), -1)
        output = self.FullyConnectedLayer(output)
        
        return output

# %%
def write_accuracy(message, file_path):
    with open(file_path, 'a') as file:
        file.write(message)
        file.write('\n')
    

# %%
num_classes = 2
learning_rate = 0.001
num_epochs = 20

model = ResidualNeuralNetwork(ResidualBlock, [2,2,2,2], num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# %%
def ValidationBlock(model):
    
    valid_set = pd.read_parquet('valid_set.parquet', dtype_backend='pyarrow')
    valid_dataset = []
    for i in range(len(valid_set)):
        valid_dataset.append((torch.tensor(valid_set['X_jets'][i]), valid_set['y'][i]))

    batch_size = 16
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
   
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            
            # resize images
            resize_transform = transforms.Compose([transforms.Resize((224, 224))])
            images = resize_transform(images)
            
            # move to device
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('Validation accuracy: {} %'.format(accuracy))
        write_accuracy('Validation accuracy: {} %'.format(accuracy), 'accuracies.txt')
    return model
        

# %%
import gc
def TrainBlock(PatchList, model):
    
    for epoch in range(num_epochs):
        batch = 1
        for patch in PatchList:
            df = pd.read_parquet(patch, dtype_backend='pyarrow')
            dataset = []
            for i in range(len(df)):
                dataset.append((torch.tensor(df['X_jets'][i]), df['y'][i]))
            
            batch_size = 16
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            dataset = None
            # Train the model
            for images, labels in train_loader:
                
                # resize images
                resize_transform = transforms.Compose([transforms.Resize((224, 224))])
                arr = [resize_transform(image) for image in images]
                
                # convert to tensor
                images = torch.stack(arr)
                
                # move to GPU
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, torch.tensor(labels, dtype=torch.long))
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del images, labels, outputs
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch {batch} Completed")
                batch += 1
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        write_accuracy(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}", 'accuracies.txt')
        # Validation
        ValidationBlock(model)
    
    return model
                
        

# %%
def TestBlock(PatchList):
    
    correct = 0
    total = 0
    for patch in PatchList:
        df = pd.read_parquet(patch, dtype_backend='pyarrow')
        dataset = []
        for i in range(len(df)):
            dataset.append((torch.tensor(df['X_jets'][i]), df['y'][i]))
        
        batch_size = 16
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        dataset = None
        with torch.no_grad():
            for images, labels in test_loader:
                
                # resize images
                resize_transform = transforms.Compose([transforms.Resize((224, 224))])
                images = resize_transform(images)
                
                # move to device
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Test accuracy: {} %'.format(accuracy))
    write_accuracy('Test accuracy: {} %'.format(accuracy), 'accuracies.txt')
    return model

# %%
PatchList = ['set0.parquet', 'set1.parquet', 'set2.parquet', 'set3.parquet', 'set4.parquet', 'set5.parquet', 'set6.parquet', 'set7.parquet', 'set8.parquet', 'set9.parquet','set10.parquet']
model = TrainBlock(PatchList, model)

# %%
model = TestBlock(['test_set1.parquet', 'test_set2.parquet', 'test_set3.parquet'])


# %%
# Save Model
torch.save(model.state_dict(), 'Resnet34Layer.pth')



