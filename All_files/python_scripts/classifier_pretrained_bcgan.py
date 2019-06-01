#!/usr/bin/env python
# coding: utf-8

# # Try a Deep Convolutional Classifier on DDSM Dataset
# 
# # Real & BC GAN Augmented data

# 
# In[1]:


from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torch.autograd import Variable
from PIL import Image

import itertools

from sklearn.metrics import roc_auc_score,  roc_curve, auc, average_precision_score, f1_score, accuracy_score

import torchvision.models as models


# In[2]:


# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# # Hyperparameters



# Root directory for dataset
dataroot = '../bc_images/'

# Root directory for dataset
labelroot='../../Pytorch_models/BC/BC/new_labels/2class/'
dataroot3= '../../olgakra/Pytorch_models/BC/BC/ddsm_patches/'


dataroot7 = 'notneeded'   
dataroot8 = '../bcgan_images/' # bcgan augmented images


# Number of workers for dataloader
num_workers = 4

num_epochs = 25

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256



# Number of GPUs available. Use 0 for CPU mode.
ngpu = 2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4,5"


# In[4]:


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
}


# In[5]:


# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# # Load the data & create a DataLoader

# In[6]:


class DDSM(torch.utils.data.Dataset):             # Notice that masks are in PNG form and mammograms in JPG
    def __init__(self, root1, root2, root3,  labelr, split, transform, transform2, mode):
        self.root1 = root1
        
        self.root2 = root2
        self.root3 = root3
        self.mode = mode
        self.labelr = labelr
        #bc_images
        imagedir =os.listdir(root1)
        random.shuffle(imagedir)
        
    
        
        if self.mode == 'train':
        
            imgs = imagedir[:500]
            auglist = os.listdir(root3)
            random.shuffle(auglist)
            img_list = imgs + auglist[:10000] # real & augmented set
        
        elif self.mode =='val':
            imgs = imagedir[3000:3400]
            img_list = imgs   # without augmented dataset, real data only
        
        elif self.mode =='test':
            imgs = imagedir[3400:]
            img_list = imgs 
        
        # Create a list with image names, labels and opposite labels
        label_list = [1]*len(img_list)
        label_o_list = [0]*len(img_list)
        bc_image_list = list(zip(img_list, label_list, label_o_list))
        #print((bc_image_list))   
        
        
        
        ### Normal images
        with open(os.path.join(labelr, '{}.txt'.format(split)), 'r') as f:
            for line in f:
                if 'normal' in line:
                    image_list = (list(map(self.process_line, f.readlines())))
        
        self.new_img_list = bc_image_list + image_list   
        
        
        self.transform = transform
        self.transform2 = transform2
        
          
    def process_line(self,line):
        if 'normal' in line:
            image_name, label = line.strip().split(' ')
            label = int(label)
           
            label_opposite = 1
            #print(image_name)
            return image_name, label, label_opposite 



    def __len__(self):
        #print(len(self.new_img_list))
        
        return len(self.new_img_list)

    def __getitem__(self, idx):
        image_name, label, label_oppos = self.new_img_list[idx]
        if 'normal' in image_name:

            image = Image.open(os.path.join(self.root2, image_name))
            image = self.transform(image)
        elif 'cancer' in image_name:
            image = Image.open(os.path.join(self.root1, image_name))
            image = self.transform2(image)
            
        else:
            image = Image.open(os.path.join(self.root3, image_name))
            image = self.transform2(image)

            
        
        return image, label, label_oppos
    


transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([256,256]),
        transforms.CenterCrop([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.4374], [0.2085])
])

transform2 = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([256,256]),
        transforms.CenterCrop([224,224]),
        transforms.ColorJitter(contrast=1.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4374], [0.2085])
])




classes = ('0', '1')



dataset = DDSM(dataroot, dataroot3,dataroot8, labelroot, 'train', transform=transform,       # use train real healthy and BC
               transform2 = transform2, mode = 'train') 



train_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size= batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=True)


vdataset = DDSM(dataroot, dataroot3, dataroot7, labelroot, 'val', transform=transform,       # use train real healthy and BC
               transform2 = transform2, mode = 'val') 



val_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size= batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=True)





feature_extract = True

model = models.resnet50(pretrained=True)
params_to_update = model.parameters()
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    
    
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(2048, 2)
optimizer_ft = optim.Adam(params_to_update, lr=10e-05)
criterion = nn.CrossEntropyLoss()
model = model.to(device)
model = model.to(device)
if (device.type == 'cuda') and ngpu > 1:
    model = nn.DataParallel(model)
    
path= '../saved_models/classifier/'



min_val_loss = np.Inf
losses, val_losses = [], []
for epoch in range(num_epochs):  

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer_ft.zero_grad()
        
            outputs = model(inputs)



            loss = criterion(outputs, labels)

            loss.backward()
            optimizer_ft.step()
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches (in which every batch pools 100 images (100*100))
                print('[%d, %5d] loss: %.3f ' %
                      (epoch + 1, i + 1, running_loss / 100))      
        
            losses.append(running_loss)
            running_loss = 0.0
        
            
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(val_loader,0):
            
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_losses.append(val_loss)
                 


                # Find the best model based on the best validation loss
                if val_loss < min_val_loss:
                # Save the model
                    torch.save(model.state_dict(), path + 'bestbc.pth') 
                    min_val_loss = val_loss
                    
                val_loss = 0
        model.train()
      
    
    

    
print('Finished Training & Validation')    
    
        
plt.figure(figsize=(10,5))
fig = plt.figure()        
plt.plot(losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend(frameon=False)
fig.savefig('ValTrainbc.png')