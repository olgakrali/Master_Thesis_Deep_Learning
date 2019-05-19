#!/usr/bin/env python
# coding: utf-8

# # Try a Deep Convolutional Classifier on DDSM Dataset
# 
# # Train the classifiers separately

# ### Add  loss and accuracy graphs

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

from sklearn.metrics import roc_auc_score,  roc_curve, auc, average_precision_score, f1_score

import torchvision.models as models


# In[2]:


# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# # Hyperparameters

# In[3]:


# Root directory for dataset
dataroot = '../bc_images/'
dataroot2 = '../bc_masks/'

# Root directory for dataset
labelroot='../new_labels/2class/'
dataroot3= '../ddsm_patches/'


dataroot4='../augmented_train_masks/'   # you can generate more masks for training another time
dataroot5='../augmented_test_masks/'

dataroot6 = '../gan_generated_images/val_gen_images/'  # 12260 images
dataroot7 = '../gan_generated_images/test_gen_images/' # 11771 images
dataroot8 = '../gan_generated_images/train_gen_images/' # 94327


# Number of workers for dataloader
num_workers = 16

num_epochs = 500

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256



# Number of GPUs available. Use 0 for CPU mode.
ngpu = 2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


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
    def __init__(self, root1, root2,  labelr, split, transform, transform2):
        self.root1 = root1
        
        self.root2 = root2
        
        self.labelr = labelr
        
        #bc_images
        imagedir =os.listdir(root1)
    
       
        
        img_list = sorted(imagedir)
                    
        
        
        
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
            image = self.transform2(image)
            
        else:
            image = Image.open(os.path.join(self.root1, image_name))
            image = self.transform(image)

            
        
        return image, label, label_oppos
    


# In[7]:


transform = transforms.Compose([
                
                transforms.Resize([256, 256]),
                transforms.ColorJitter(contrast=1.5),
                transforms.ToTensor(),
                
                transforms.Normalize(mean=[0.5], std=[0.5])])

transform2 = transforms.Compose([
                
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
                
                transforms.Normalize(mean=[0.5], std=[0.5])])





classes = ('0', '1')


# Generated BC images  + Real Healthy

# In[10]:


# Use the train generated images, thus we get a balanced dataset
gen_dataset = DDSM(dataroot8, dataroot3, labelroot, 'train', transform=transform,
               transform2 = transform2)



gentrain_loader = torch.utils.data.DataLoader(gen_dataset,
                                        batch_size= batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=True)


# In[11]:


gen_tdataset = DDSM(dataroot7, dataroot3, labelroot, 'test', transform=transform,
               transform2 = transform2)

gen_test_loader = torch.utils.data.DataLoader(gen_tdataset,
                                        batch_size= batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=True)


# # Try other models (Try without image transformation and pretrained = True)

# # 1) Resnet

# # a) Resnet 50

# In[12]:



resnet = models.resnet50() # Pretrained did not work properly here
        
state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50'],model_dir ='~/.cache/torch')
resnet.load_state_dict(state_dict)


# In[13]:


resnet.fc = nn.Linear(2048, 2)
resnet = resnet.to(device)
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 2)
model = model.to(device)


# # Weight initialization by utilizing pretrained weights

# In[17]:


pretrained_dict = resnet.state_dict() 
model_dict = model.state_dict() 
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 

# update & load
model_dict.update(pretrained_dict) 
model.load_state_dict(model_dict)




# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.CrossEntropyLoss()
# Handle multi-gpu if desired
model = model.to(device)
if (device.type == 'cuda') and ngpu > 1:
    model = nn.DataParallel(model)





# # Real + Generated images

# In[ ]:


glosses = []
gacc = []

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(gentrain_loader, 0):
        # get the inputs
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ft.step()

        # print statistics
        #running_loss += loss.item()
        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()

        if i % 100 == 99:    # print every 100 mini-batches (in which every batch pools 100 images (100*100))
            print('[%d, %5d] loss: %.3f Accuracy: %.2f' %
                  (epoch + 1, i + 1, loss.item(), accuracy))
            running_loss = 0.0
            
        glosses.append(loss.item())
        gacc.append(accuracy)

print('Finished Training')


# In[ ]:


# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for i, data in enumerate(gen_test_loader, 0):
        # get the inputs
        images, labels = data[0].to(device), data[1].to(device)
    

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy of the model on real and generated images: {} %'.format(100 * correct / total))


# In[ ]:


r18gdataiter = iter(gen_test_loader)
r18gimages, r18glabels = r18gdataiter.next()[0], r18gdataiter.next()[1] 

# print images
#imshow(torchvision.utils.make_grid(images))
#plt.imshow(np.transpose(vutils.make_grid(r18gimages.to(device), padding=2, normalize=True).cpu(),(1,2,0)))
#print('GroundTruth: ', ' '.join('%5s' % classes[r18glabels[j]] for j in range(64)))


# In[ ]:


res18goutputs = model(r18gimages.to(device))


# In[ ]:


_, res18gpredicted = torch.max(res18goutputs, 1)

#print('Predicted: ', ' '.join('%5s' % classes[res18gpredicted[j]]
                              #for j in range(64)))


# In[ ]:


r18roc_score_g =roc_auc_score(r18glabels.cpu(), res18gpredicted.cpu())
r18average_precision_g = average_precision_score(r18glabels.cpu(),res18gpredicted.cpu() )
r18F1_g = f1_score(r18glabels.cpu(), res18gpredicted.cpu(), average='weighted')


# In[ ]:


print('ROC AUC:', r18roc_score_g, 'Average precision:', r18average_precision_g, 'F1 score:', r18F1_g)


# In[ ]:


num_classes = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in classes:
    fpr[i], tpr[i], _ = roc_curve(r18glabels.cpu(), res18gpredicted.cpu())
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(r18glabels.cpu().numpy().ravel(), res18gpredicted.cpu().numpy().ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[ ]:


plt.figure()
lw = 2
plt.plot(fpr['0'], tpr['0'], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc['0'])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Real & Generated Images in Resnet50')
plt.legend(loc="lower right")
#plt.show()
fig.savefig('Classification_loss_Gen.png')

# In[ ]:


plt.figure(figsize=(10,5))
fig = plt.figure()
plt.title("Classification Loss During Training")
plt.plot(glosses, color = 'm',label="Classification Loss Real Healthy and BC generated images")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
#plt.show()
fig.savefig('Classification_loss_Gen.png')


# In[ ]:


plt.figure(figsize=(10,5))
fig = plt.figure()
plt.title("Classification Accuracy During Training")
plt.plot(gacc, color = 'm', label="Classfication Accuracy Real Healthy and BC generated images")
plt.xlabel("iterations")
plt.ylabel("Accuracy")
plt.legend()
#plt.show()
fig.savefig('Classification_accuracy_Gen.png')




