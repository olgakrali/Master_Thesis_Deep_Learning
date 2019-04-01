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
dataroot='/clio/users/olgakr/DDSM/new_labels/2class/'
dataroot2= '/clio/users/olgakr/DDSM/ddsm_patches/'

dataroot3 = '../ddsm_masks/train_masks/'  # mask path

# Number of workers for dataloader
num_workers = 36

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
num_epochs = 100


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


# Mask Data Loader
class MASKS(torch.utils.data.Dataset):
    def __init__(self, root,transform):
        self.root = root
        self.image_list = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = Image.open(os.path.join(self.root,image_name))
    
        image = self.transform(image)
        
        return image
    
    
transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                               transforms.Resize([64,64]), 
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5], std=[0.5])
                           ])
mask_dataset = MASKS(dataroot3, transform=transform)

mask_loader = torch.utils.data.DataLoader(
        mask_dataset, batch_size=64, shuffle=True,
        num_workers=10, pin_memory=True)

mask_patch = next(iter(mask_loader))
# set masks from 0-1
np.place(mask_patch.numpy(),mask_patch.numpy()>-1, [1])
np.place(mask_patch.numpy(),mask_patch.numpy()==-1, [0])


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
    
    # transform real images to have a noise input
    
    for index in indices:
        realdata[int(index[0])][int(index[1])][int(index[2])][int(index[3])] = np.random.uniform(-1,1)
    
    return realdata


# # Load the data & create a DataLoader

# In[4]:


class DDSM(torch.utils.data.Dataset):
    def __init__(self, root1, root2, split, transform):
        self.root1 = root1
        self.root2 = root2
        def process_line(line):
            if 'normal' in line:
                image_name, label = line.strip().split(' ')
                label = int(label)
                return image_name, label
        with open(os.path.join(root1, '{}.txt'.format(split)), 'r') as f:
            for line in f:
                if 'normal' in line:
                    self.image_list = (list(map(process_line, f.readlines())))
            
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name, label = self.image_list[idx]
        image = Image.open(os.path.join(self.root2, image_name)) #.convert("L")
        image = self.transform(image)
        
        return image, label
    

transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize([64,64]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])])

train_dataset = DDSM(dataroot, dataroot2, 'train',  transform=transform)



train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)


# In[5]:


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(train_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:10], padding=2, normalize=True).cpu(),(1,2,0)))


# In[7]:


real_batch[0].shape # G and D should generate and be fed with 28x28 images!!


# In[8]:


torch.max(real_batch[0])


# In[9]:


torch.min(real_batch[0])


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

class encoder(nn.Module):
    def __init__(self, ngpu):
        super(encoder, self).__init__()
        self.ngpu = ngpu
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
    
class decoder(nn.Module):
    def __init__(self, ngpu):
        super(decoder, self).__init__()
        self.ngpu = ngpu
            
        self.dec1 = nn.ConvTranspose2d((ndf*16)*2, ngf * 8, 4, 2, 1, bias=False)
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
     
    def forward(self, input, extranoise):
        x = torch.cat([input, extranoise], 1)
        x = F.relu(self.bn1(self.dec1(x)))
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


# Create the encoder
encoder = encoder(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    encoder = nn.DataParallel(encoder, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
encoder.apply(weights_init)

# Print the model
print(encoder)


# Create the decoder
decoder = decoder(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    decoder = nn.DataParallel(decoder, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
decoder.apply(weights_init)

# Print the model
print(decoder)

# # Class Discriminator

# In[13]:

#Linear projection of Gaussian distribution


class Projection(nn.Module):
    def __init__(self, ngpu):
        super(Projection, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nz, ndf*8),
            nn.ReLU(True),
            nn.Linear(ndf*8, ndf*16*4*4),
            nn.Tanh())
    def forward(self, input):
        return self.main(input)
    
    
netp = Projection(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netp = nn.DataParallel(netp, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netp.apply(weights_init)

# Print the model
print(netp)

p_optimizer = torch.optim.Adam(netp.parameters(), lr=0.0002)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
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
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)


# In[15]:

#Loss

adversarial_loss = torch.nn.BCELoss()
adversarial_loss = adversarial_loss.to(device)
pixelwise_loss = torch.nn.L1Loss()
pixelwise_loss = pixelwise_loss.to(device)


# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(beta1, 0.999))


# In[17]:


# Establish convention for real and fake labels during training
fixed_noise = createpatch(real_batch[0], mask_patch, batch_size)
fixed_noise = fixed_noise.to(device)

extra_noise = torch.randn(64, nz).to(device)

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

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (data,mask) in enumerate(zip((train_loader),(mask_loader))):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
    
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        
        np.place(mask.numpy(),mask.numpy()>-1, [1])
        np.place(mask.numpy(),mask.numpy()==-1, [0])
        
        #print('real:',real_cpu.shape)  # shape of the image tensor
        b_size = real_cpu.size(0)  # by using b_size instead of batch_size you get the opportunity for your last batch
                                   # to have size <64 without getting any error when you call for instance the extra
                                   #noise
        noise = createpatch(data[0], mask, b_size)
        noise = noise.to(device)
        
        mask = mask.to(device)
       
       
        
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
        
        ext_noise = torch.randn(b_size, nz).to(device=device)
        l_proj = (netp(ext_noise)).reshape(b_size, ndf*16, 4, 4)
        p_optimizer.step()
        
        encoded_imgs = encoder(noise.float(),mask.float())
        fake = decoder(encoded_imgs.float(), l_proj.float())
        
        
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
                
                hey = (netp(extra_noise)).reshape(64, ndf*16, 4, 4)
                
                fake = decoder(encoded_imgs.float(), hey.float())
                
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
fig.savefig('Losses-patch-AE-BoxInfilling.png')


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


vutils.save_image(real_batch[0].to(device)[:64], 'RealPatch-AE-BoxInfilling.png', padding=2, normalize=True)


# In[ ]:


vutils.save_image(img_list[-1], 'FakePatch-AE-BoxInfilling.png')

path= '../saved_models/model-DCGAN-BC-AdversarialEncDec-patch-MaskInfilling-BCMasks-2chan-Noise/'

torch.save(encoder.state_dict(), path + 'encoder-patch-AE-100e.pth') #encoder model is for lr=0.2
torch.save(decoder.state_dict(), path + 'decoder-patch-AE-100e.pth') #encoder model is for lr=0.2

torch.save(netD.state_dict(), path + 'netD-patch-AE-100e.pth')


