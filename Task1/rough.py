#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:







# In[1]:


import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[2]:


# reading the data in hdf5
import h5py

# open the file
f = h5py.File('SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5', 'r')


# In[3]:


# set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[4]:


# Convert to tensor
electron_images = torch.tensor(f['X'][:])
electron_labels = torch.tensor(f['y'][:])


# In[5]:


# Photons
f = h5py.File('SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5', 'r')


# In[6]:


# Convert to tensor
photon_images = torch.tensor(f['X'][:])
photon_labels = torch.tensor(f['y'][:])


# In[ ]:





# In[7]:


from torch.utils.data import TensorDataset
electron_set = TensorDataset(electron_images, electron_labels)
photon_set = TensorDataset(photon_images, photon_labels)

electron = []
for i in range(len(electron_set)):
    electron.append((electron_set[i][0], electron_set[i][1].item()))

photon = []
for i in range(len(photon_set)):
    photon.append((photon_set[i][0], photon_set[i][1].item()))
    
# take 1 from each class
dataset = []
for i in range(len(electron)):
    dataset.append(electron[i])
    dataset.append(photon[i])

# for i in range(len(electron)):
#     dataset.append(electron[i])
# for i in range(len(photon)):
#     dataset.append(photon[i])

electron = None
photon = None


# In[99]:


# Concatenate the images
# images = torch.cat((electron_images, photon_images))
# labels = torch.cat((electron_labels, photon_labels))


# In[100]:


# # Combine Images and Labels into a single dataset
# from torch.utils.data import TensorDataset
# dataset = TensorDataset(images, labels)
# dataset1 = []


# In[101]:


# for i in range(len(dataset)):
#     dataset1.append([dataset[i][0], dataset[i][1].item()])

# dataset = dataset1
# dataset1 = None


# In[8]:


indices = list(range(len(dataset)))
np.random.shuffle(indices)
dataset = [dataset[i] for i in indices]


# In[9]:


# Splitting the dataset using random_split
from torch.utils.data import random_split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print(len(train_dataset), len(test_dataset))


# In[104]:


print(train_dataset[0][0].shape)


# In[105]:


# Further break training data into train/valid sets
train_size = int(0.85 * len(train_dataset))
valid_size = len(train_dataset) - train_size

train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
print(len(train_dataset), len(valid_dataset), len(test_dataset))


# In[106]:


#DataLoader
from torch.utils.data import DataLoader
batch_size = 100
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


# In[107]:


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


# In[108]:


class ResidualNeuralNetwork(nn.Module):
    
    def __init__(self, block, layers, num_classes=10):
        super(ResidualNeuralNetwork, self).__init__()
        self.inplanes = 64
        
        # Convolutional layers
        
        # in_channels = 3 as we have RGB images
        # First Convolution Layer with 64 Filters of size 7x7, stride=2 and padding=3
        self.convolutionLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=3),
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


# In[109]:


device


# In[110]:


num_classes = 2
learning_rate = 0.01
num_epochs = 15

# Initialize the model
model = ResidualNeuralNetwork(ResidualBlock, [3, 4, 6, 3], num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# In[ ]:









# In[111]:


# training the model
total_step = len(train_loader)
import torch.nn.functional as F
import gc
import sys

def write_accuracies(message,file_path):
    with open(file_path, 'a') as file:
        file.write(message)
        file.write('\n')

iter = 0
for epoch in range(num_epochs):
    batch = 1
    for i, (images, labels) in enumerate(train_loader):
        
        # permute the image dimensions

        number = len(images)
        arr = [torch.tensor(i).permute(2,0,1) for i in images]

        images = torch.stack(arr)
        
        # resize the image to 224x224
        resize_transform = transforms.Resize((224,224))
        arr = [resize_transform(i) for i in images]


        images = torch.stack(arr)
    
    
        # move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        outputss = F.softmax(outputs, dim=1)
        loss = criterion(outputs, torch.tensor(labels, dtype=torch.long))
        
        # Backward and optimize
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        # particular batch number of particular epoch is completed
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch {batch} completed")
        batch += 1
        

    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    write_accuracies(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}", "accuracies.txt")
    

    # Validation accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        
        for image, labels in valid_loader:
            
            # permute the image dimensions

            number = len(image)
            arr = [torch.tensor(i).permute(2,0,1) for i in image]

            image = torch.stack(arr)
            
            # resize the image to 224x224
            resize_transform = transforms.Resize((224,224))
            arr = [resize_transform(i) for i in image]


            image = torch.stack(arr)
             
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('Accuracy of the model on the validation set: {} %'.format(100 * correct / total))
        write_accuracies(f'Accuracy of the model on the validation: {100 * correct / total} %', "accuracies.txt")


# In[112]:


#import F
from torch.nn import functional as F
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        
        # permute the image dimensions

        number = len(images)
        arr = [torch.tensor(i).permute(2,0,1) for i in images]

        images = torch.stack(arr)
        
        
        # resize the image to 224x224
        target_height = target_width = 224
        arr = []
        
        for image_tensor in images:
            # padding_height = max(0, target_height - image_tensor.shape[1])  # Padding needed in the height dimension
            # padding_width = max(0, target_width - image_tensor.shape[2])  # Padding needed in the width dimension

            # # Pad the image tensor
            # padded_image = F.pad(image_tensor, (0, padding_width, 0, padding_height), mode='constant', value=0)  
            # arr.append(padded_image)
            
            # resize to 224
            resize_transform = transforms.Resize((224,224))
            arr.append(padded_image)

        images = torch.stack(arr)

        
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    write_accuracies(f'Test Accuracy of the model on the test images: {100 * correct / total} %', "accuracies.txt")



# In[113]:


# Saving the model
torch.save(model.state_dict(), 'resnet.pth')


# In[ ]:









# In[ ]:






