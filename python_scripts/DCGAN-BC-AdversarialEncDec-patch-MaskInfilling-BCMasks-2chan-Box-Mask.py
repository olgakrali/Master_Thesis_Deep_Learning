#!/usr/bin/env python
# coding: utf-8

# # Try a Deep Convolutional Generative Adversarial Network on DDSM Dataset

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
labelroot='/clio/users/olgakr/DDSM/new_labels/2class/'
dataroot= '/clio/users/olgakr/DDSM/ddsm_patches/'
dataroot2='../ddsm_masks/train_masks/'

dataroot3='../ddsm_masks/val_masks/'
dataroot4='../ddsm_masks/test_masks/'
# Number of workers for dataloader
num_workers = 12

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 50


# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")





#This function adds the mask as a noise input on the mammogram data (without modifying the original images)


def createpatch(mydata, masks, batch_size):
    real = Variable(mydata.data.clone())
    mask = Variable(masks.data.clone())
    realdata = real[:batch_size]
    maskdata = mask[:batch_size]
    
    ## find the mask location
    maskpos = np.where(maskdata==1)
    
    #create index file
    indices = np.empty((maskpos[0].shape[0], len(maskpos)))
    for i in range(maskpos[0].shape[0]):
    
        indices[i] = np.array((maskpos[0][i],maskpos[1][i],maskpos[2][i], maskpos[3][i]))
    
    
    
    #Get the min max index for rows and columns of all images and obtain the biggest sizes
    #print(indices[1])
    
    
    # transform real images to have a noise input
    
    for index in indices:
        realdata[int(index[0])][int(index[1])][int(index[2])][int(index[3])] = np.random.uniform(-1,1)
    
    count = 0
    rowlist = []
    columnlist = []
    size_list = []
    index_list = []
    new_batch = np.append(maskpos[0],64) # add an extra  random number to make the for loop work for all 64 batches
    new_rows = np.append(maskpos[2],64)
    new_columns = np.append(maskpos[3],64)
    for batch, row, column in zip (new_batch, new_rows, new_columns):
        
        
        if batch == count:
            rowlist.append(row)
            #print(min(rowlist))
            columnlist.append(column)
            count = count + 0
            
    
                
        
        else:     # do calculations and empty the list to create new with the new batch
            #print('Rows', max(rowlist), min(rowlist))
            #print('Columns', max(columnlist)-min(columnlist))
            if all ([len(rowlist) != 0, len(columnlist)!=0]):
                index_list.append([min(rowlist), max(rowlist),min(columnlist), max(columnlist)])
                size_list.append([max(rowlist)-min(rowlist),max(columnlist)-min(columnlist)])

                rowlist.clear()
                columnlist.clear()
                count = count + 1
           
            else:
                index_list.append([0,0,0,0])  # For those mask images that surprisingly stayed out of all-black filtering
                size_list.append([0,0])
                count = count + 1
                pass
                
            
    realdata2 = np.copy(realdata)
    
    for i, (index) in enumerate(index_list): #zip does not work due to float and int in the same call function
        for j in range(index[0],index[1]+1):
            for k in range(index[2],index[3]+1):
                realdata2[i][:, j, k] = np.random.uniform(-1,1)
        
    realdata2 = torch.from_numpy(realdata2)
     
    #print(len(size_list))
    
  #  print('Get the largest box size: ', max(size_list))    # Tha doesn't work properly
    
    return realdata, size_list, index_list, realdata2

# # Load the data & create a DataLoader

# In[4]:


class DDSM(torch.utils.data.Dataset):             # Notice that masks are in PNG form and mammograms in JPG
    def __init__(self, root1, root2, root3, split, transform, transform2):
        self.root1 = root1
        self.root2 = root2
        self.root3 = root3
        
        with open(os.path.join(root3, '{}.txt'.format(split)), 'r') as f:
            for line in f:
                if 'normal' in line:
                    self.image_list = (list(map(self.process_line, f.readlines())))
        
                    
        self.mask_list = os.listdir(root2)           
        self.transform = transform
        self.transform2 = transform2
        
          
    def process_line(self,line):
        if 'normal' in line:
            image_name, label = line.strip().split(' ')
            label = int(label)
            #print(image_name)
            return image_name, label    

    def __len__(self):
        
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name,label = self.image_list[idx]

        image = Image.open(os.path.join(self.root1, image_name)).convert("L")
        image = self.transform(image)
        
        mask_name = self.mask_list[idx]
        mask = Image.open(os.path.join(self.root2,mask_name)).convert("L")
    
        mask = self.transform2(mask) 
        
        return image, mask
    

transform = transforms.Compose([
                
                transforms.Resize([64,64]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])])

transform2 = transforms.Compose([
             
                transforms.Resize([64,64]),
                transforms.ToTensor()])

train_dataset = DDSM(dataroot, dataroot2, labelroot, 'train', transform=transform, transform2 = transform2)



train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)


# In[5]:



#  training images
real_batch = next(iter(train_loader))

# patches
mask_patch = real_batch[1]
np.place(mask_patch.numpy(),mask_patch.numpy()>0, [1])
print(torch.min(mask_patch))
print(torch.max(mask_patch))


# #  Weight initialization for both Generator and Discriminator

# In[10]:


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# # Class Adversarial AutoEncoder Generator

# In[11]:


# Generator Code

# Generator Code

# Generator Code

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #input 1x64x64
        self.conv1 = nn.Conv2d(2, ngf, 3, 1, 1, bias=False) # one channel for the image one for the mask
        self.bn1 = nn.BatchNorm2d(ngf)   
        #first layer ngf x 64 x 64
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False)  # second layer will be concatenated (remember x and y)
        self.bn2 = nn.BatchNorm2d(ngf * 2)
        #second layer ngf*2 x 32 x 32
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 4)
        #third layer ngf*4 x 16 x 16
        self.conv4 =  nn.Conv2d( ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf * 8)
        #forth layer ngf*8 x 8 x 8
        self.conv5 =  nn.Conv2d( ngf * 8, ngf * 16, 4, 2, 1, bias=False)
        
        #fifth layer ngf*16 x 4 x 4
            
    def forward(self, input, masking):
        x = torch.cat([input, masking], 1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        #print('1: ', x.shape)
        x = F.leaky_relu(self.bn2(self.conv2(x)),0.2)
        #print('2: ', x.shape)
        x = F.leaky_relu(self.bn3(self.conv3(x)),0.2)
        #print('3: ', x.shape)
        x = F.leaky_relu(self.bn4(self.conv4(x)),0.2)
        #print('4: ', x.shape)
        x = torch.tanh(self.conv5(x))
        #print('5: ', x.shape)
        return x
            
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
            
        self.dec1 = nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)
        #1st layer ngf* 8 x 8 x 8
        self.dec2 = nn.ConvTranspose2d(ngf*8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)
        #2nd layer ngf*4 x 16 x 16
        self.dec3 = nn.ConvTranspose2d(ngf*4, ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)
        #3rd ngf*2 x 32 x 32
        self.dec4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        #4th layer ngf x 64 x 64
        self.dec5 = nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False)
        #5th layer 1 x 64 x 64
     
    def forward(self, input):
        x = F.relu(self.bn1(self.dec1(input)))
        #print('1: ', x.shape)
        x = F.relu(self.bn2(self.dec2(x)))
        #print('2: ', x.shape)
        x = F.relu(self.bn3(self.dec3(x)))
        #print('3: ', x.shape)
        x = F.relu(self.bn4(self.dec4(x)))
        #print('4: ', x.shape)
        x = torch.tanh(self.dec5(x))
        #print('5: ', x.shape)

        return x


# In[12]:


encoder = Encoder()
encoder = encoder.to(device)


encoder.apply(weights_init)

# Handle multi-gpu if desired
if (device.type == 'cuda') and ngpu > 1:
#     encoder = nn.DataParallel(encoder, list(range(ngpu))) ### ti einai auto pou kaneis edw de katalava..!
    encoder = nn.DataParallel(encoder)
    
print(encoder.module)


# Create the decoder
decoder = Decoder()
decoder = decoder.to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
decoder.apply(weights_init)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    decoder = nn.DataParallel(decoder)


# Print the model
print(decoder.module)


# # Class Discriminator

# In[13]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
  
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64   
            nn.Conv2d(nc, ndf , 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf ) x 32 x 32
            nn.Conv2d(ndf , ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 8) x 4 x 4
            nn.Conv2d(ndf *8 , 1, 4, 1, 0, bias=False),  
            nn.Sigmoid()
            # prob (0,1)
        )

    def forward(self, input):
        return self.main(input)

# In[14]:


# Create the Discriminator
netD = Discriminator()
netD = netD.to(device)


# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD) 

# Print the model
print(netD.module)


# In[15]:

#Loss

adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()



# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(beta1, 0.999))


# In[17]:


# Establish convention for real and fake labels during training
imgs, mylist, indexlist,fixed_noise = createpatch(real_batch[0], mask_patch, batch_size)

fixed_noise = fixed_noise.to(device)



real_label = 1
fake_label = 0
fake_labels = torch.zeros(batch_size, 1)


label = torch.FloatTensor(batch_size).to(device)

# In[18]:


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (data) in enumerate(train_loader,0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
    
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        np.place(data[1].numpy(),data[1].numpy()>0, [1])
        mask_cpu = data[1].to(device)
       
        
        #print('real:',real_cpu.shape)  # shape of the image tensor
        b_size = real_cpu.size(0)
        images, size, index,noise = createpatch(data[0], data[1], b_size)
        noise = noise.to(device)
        
        
       
       
        
        label = torch.full((b_size,), real_label).to(device)
        #print('label shape:', label.shape)  # shape of the real label tensor
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1) 
        #print(output)
          
        #print('output shape:', output.shape) # shape of the generated label tensor
        # Calculate loss on all-real batch
        errD_real = adversarial_loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()  # the output of real images has to be close to 1 (prob for real close to 1)

        
        
        
        ## Train with all-fake batch
        # Generate batch of latent vectors
        
        # Generate fake image batch with G
        
       
        
        encoded_imgs = encoder(noise.float(),mask_cpu.float())
        fake = decoder(encoded_imgs)
        
        
        #print('fake:',fake.shape)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        #print(label.shape)
        #print(output.shape)
        # Calculate D's loss on the all-fake batch
        errD_fake = adversarial_loss(output, label)  ### Label = 0 since it's fake, if D is good output should also be around 0
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()   # the output of fake images has to be close to 0 (prob for fake close to 0)
        
        # Add the gradients from the all-real and all-fake batches
        errD = 0.5 * (errD_real + errD_fake)  ## a high score means a good discriminator (look on the DmaxGmin equation)
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        encoder.zero_grad()
        decoder.zero_grad()
        
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = 0.001* (adversarial_loss(output, label)) + 0.999 * (pixelwise_loss(fake, real_cpu))
        
        # Calculate gradients for G
        #reset_grad() do I need this?
        errG.backward()
        D_G_z2 = output.mean().item()  
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch+1, num_epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on generated images
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                encoded_imgs = encoder(fixed_noise.float(), mask_patch.to(device).float())
                fake = decoder(encoded_imgs)
            img_list.append(vutils.make_grid(fake.detach().cpu(), padding=2, normalize=True))

        iters += 1



# In[16]:


plt.figure(figsize=(10,5))
fig = plt.figure()
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
#plt.show()
fig.savefig('Losses-patch-AE-Box-Mask.png')


# In[ ]:



# Plot the real images
#plt.figure(figsize=(15,15))
#plt.subplot(1,2,1)
#plt.axis("off")
#plt.title("Real Images")
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
#plt.subplot(1,2,2)
#plt.axis("off")
#plt.title("Fake Images")
#plt.imshow(np.transpose(img_list[-1],(1,2,0)))  
#plt.show()


# In[17]:


vutils.save_image(real_batch[0].to(device)[:64], 'RealPatch-AE-Box-Mask.png', padding=2, normalize=True)


# In[ ]:


vutils.save_image(img_list[-1], 'FakePatch-AE-Box-Mask.png')

path= '../saved_models/model-DCGAN-BC-AdversarialEncDec-patch-MaskInfilling-BCMasks-2chan-Box-Mask/'

torch.save(encoder.state_dict(), path + 'encoder-patch-AE-50e.pth') #encoder model is for lr=0.2
torch.save(decoder.state_dict(), path + 'decoder-patch-AE-50e.pth') #encoder model is for lr=0.2

torch.save(netD.state_dict(), path + 'netD-patch-AE-50e.pth')


