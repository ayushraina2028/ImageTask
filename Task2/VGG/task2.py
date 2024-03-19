#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[3]:


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[4]:




# In[5]:





# In[6]:



# In[7]:


class VGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()
        
        # Convolutional layers
        
        #Layer 1 -> 3 Input channels, 64 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization and ReLU activation
        self.convolutionalLayer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.ReLU())
        
        #Layer 2 -> 64 Input channels, 64 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization, Relu activation and max pooling
        self.convolutionalLayer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2))
        
        #Layer 3 => 64 Input channels, 128 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization and ReLU activation
        self.convolutionalLayer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU())
        
        #Layer 4 => 128 Input channels, 128 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization, Relu activation and max pooling
        self.convolutionalLayer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2))
        
        #Layer 5 => 128 Input channels, 256 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization and ReLU activation
        self.convolutionalLayer5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU())
        
        #Layer 6 => 256 Input channels, 256 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization and ReLU activation
        self.convolutionalLayer6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU())
        
        #Layer 7 => 256 Input channels, 256 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization, Relu activation and max pooling
        self.convolutionalLayer7 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2))
        
        #Layer 8 => 256 Input channels, 512 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization and ReLU activation
        self.convolutionalLayer8 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU())
        
        #Layer 9 => 512 Input channels, 512 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization and ReLU activation
        self.convolutionalLayer9 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU())
        
        #Layer 10 => 512 Input channels, 512 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization, Relu activation and max pooling
        self.convolutionalLayer10 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2))
        
        #Layer 11 => 512 Input channels, 512 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization and ReLU activation
        self.convolutionalLayer11 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU())
        
        #Layer 12 => 512 Input channels, 512 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization and ReLU activation
        self.convolutionalLayer12 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU())
        
        #Layer 13 => 512 Input channels, 512 output channels, 3x3 kernel size, stride of 1 and padding of 1, followed by normalization, Relu activation and max pooling
        self.convolutionalLayer13 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(512),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Fully connected layers
        
        # Flatten the output of the last convolutional layer
        self.fullyConnectedLayer1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(7*7*512, 4096), nn.ReLU())
        
        # Fully connected layer 2
        self.fullyConnectedLayer2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU())
        
        # Fully connected layer 3
        self.fullyConnectedLayer3 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        
        output = self.convolutionalLayer1(x)
        output = self.convolutionalLayer2(output)
        output = self.convolutionalLayer3(output)
        output = self.convolutionalLayer4(output)
        output = self.convolutionalLayer5(output)
        output = self.convolutionalLayer6(output)
        output = self.convolutionalLayer7(output)
        output = self.convolutionalLayer8(output)
        output = self.convolutionalLayer9(output)
        output = self.convolutionalLayer10(output)
        output = self.convolutionalLayer11(output)
        output = self.convolutionalLayer12(output)
        output = self.convolutionalLayer13(output)
        output = output.reshape(output.size(0), -1)
        output = self.fullyConnectedLayer1(output)
        output = self.fullyConnectedLayer2(output)
        output = self.fullyConnectedLayer3(output)
        
        return output


# In[8]:


def write_accuracy(message, file_path):
    with open(file_path, 'a') as file:
        file.write(message)
        file.write('\n')
    


# In[9]:


num_classes = 2
learning_rate = 0.001
num_epochs = 15

model = VGG16(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# In[10]:


def ValidationBlock(model):
    
    valid_set = pd.read_parquet('valid_set.parquet', dtype_backend='pyarrow')
    valid_dataset = []
    for i in range(len(valid_set)):
        valid_dataset.append((torch.tensor(valid_set['X_jets'][i]), valid_set['y'][i]))

    batch_size = 16
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    df = None
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
        


# In[11]:


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
                
        


# In[12]:


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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Test accuracy: {} %'.format(accuracy))
    write_accuracy('Test accuracy: {} %'.format(accuracy), 'accuracies.txt')
    return model


# In[13]:


PatchList = ['set0.parquet', 'set1.parquet', 'set2.parquet', 'set3.parquet', 'set4.parquet', 'set5.parquet', 'set6.parquet', 'set7.parquet', 'set8.parquet', 'set9.parquet','set10.parquet']
model = TrainBlock(PatchList, model)


# In[ ]:


model = TestBlock(['test_set1.parquet', 'test_set2.parquet', 'test_set3.parquet'])


# In[ ]:


# Save Model
torch.save(model.state_dict(), 'VGG16.pth')

