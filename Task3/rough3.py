#!/usr/bin/env python
# coding: utf-8

# In[14]:


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


# In[15]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[16]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[17]:


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


# In[18]:


class ResidualNeuralNetwork(nn.Module):
    
    def __init__(self, block, layers, num_classes=10):
        super(ResidualNeuralNetwork, self).__init__()
        self.inplanes = 64
        
        # Convolutional layers
        
        # in_channels = 3 as we have RGB images
        # First Convolution Layer with 64 Filters of size 7x7, stride=2 and padding=3
        self.convolutionLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=64, kernel_size=7, stride=2, padding=3),
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


# In[19]:


def write_accuracy(message, file_path):
    with open(file_path, 'a') as file:
        file.write(message)
        file.write('\n')
    


# In[20]:


num_classes = 1
learning_rate = 0.004
num_epochs = 20

model = ResidualNeuralNetwork(ResidualBlock, [3, 4, 6, 3], num_classes).to(device)

# Use MSE as loss for regression 
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# change learning rate after 7 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


# In[21]:


def ValidationBlock(model):
    
    PatchList = ['valid0.parquet', 'valid1.parquet']
    
    totalLoss = 0
    numSamples = 0
    for patch in PatchList:
        valid_set = pd.read_parquet(patch, dtype_backend='pyarrow')
        valid_dataset = []
        for i in range(len(valid_set)):
            valid_dataset.append((torch.tensor(valid_set['X_jet'][i]), valid_set['m'][i]))

        batch_size = 16
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        valid_set = None
        with torch.no_grad():

            for images, labels in valid_loader:
                
                # resize images
                resize_transform = transforms.Compose([transforms.Resize((224, 224))])
                images = resize_transform(images)
                
                # move to device
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                totalLoss += np.log(criterion(outputs, labels).item())*images.size(0)
                numSamples += images.size(0)
            
                
    meanLoss = totalLoss/numSamples
    print('Validation Loss: {:.4f}'.format(meanLoss))
    write_accuracy('Validation Loss: {:.4f}'.format(meanLoss), 'accuracies.txt')
    return model
        


# In[22]:


import gc
def TrainBlock(PatchList, model):
    
    for epoch in range(num_epochs):
        batch = 1
        runningLoss = 0
        numSamples = 0
        for patch in PatchList:
            df = pd.read_parquet(patch, dtype_backend='pyarrow')
            dataset = []
            for i in range(len(df)):
                dataset.append((torch.tensor(df['X_jet'][i]), df['m'][i]))
            
            batch_size = 16
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            df = None
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
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.squeeze(), torch.tensor(labels, dtype=torch.float))
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                runningLoss += np.log(loss.item())*images.size(0)
                numSamples += images.size(0)
                
                print(f"Batch {batch} Loss: {runningLoss}")
                print(f"Batch {batch} Samples: {numSamples}")
                del images, labels, outputs
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch {batch} Completed")
                batch += 1
        
        epochLoss = runningLoss/numSamples
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epochLoss))
        write_accuracy('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epochLoss), 'accuracies.txt')

        # Validation
        ValidationBlock(model)
    
    return model
                
        


# In[23]:


def TestBlock(PatchList):
    
    totalLoss = 0
    numSamples = 0
    mean_loss = 0
    for patch in PatchList:
        df = pd.read_parquet(patch, dtype_backend='pyarrow')
        dataset = []
        for i in range(len(df)):
            dataset.append((torch.tensor(df['X_jet'][i]), df['m'][i]))
        
        batch_size = 16
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        df = None
        with torch.no_grad():
            for images, labels in test_loader:
                
                # resize images
                resize_transform = transforms.Compose([transforms.Resize((224, 224))])
                images = resize_transform(images)
                
                # move to device
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                totalLoss += np.log(criterion(outputs, labels).item())*images.size(0)
                numSamples += images.size(0)
                
    meanLoss = totalLoss/numSamples
    print('Test Loss: {:.4f}'.format(meanLoss))
    write_accuracy('Test Loss: {:.4f}'.format(meanLoss), 'accuracies.txt')
    return model


# In[24]:


# chunk 0-10 for gun0-gun6
PatchList = ['gun0_chunk_0.parquet', 'gun0_chunk_1.parquet', 'gun0_chunk_2.parquet', 'gun0_chunk_3.parquet', 'gun0_chunk_4.parquet', 'gun0_chunk_5.parquet', 'gun0_chunk_6.parquet', 'gun0_chunk_7.parquet', 'gun0_chunk_8.parquet', 'gun0_chunk_9.parquet', 'gun0_chunk_10.parquet']
model = TrainBlock(PatchList, model)


# In[ ]:





# In[ ]:


TestList = ['test0.parquet', 'test1.parquet', 'test2.parquet']
model = TestBlock(TestList)


# In[ ]:


# Save model to disk
torch.save(model.state_dict(), 'RegressionModel.pth')

