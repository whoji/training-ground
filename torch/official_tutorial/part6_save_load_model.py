# https://pytorch.org/tutorials/beginner/saving_loading_models.html

'''
torch.save: Saves a serialized object to disk. This function uses Python’s pickle utility for serialization. Models, tensors, and dictionaries of all kinds of objects can be saved using this function.
torch.load: Uses pickle’s unpickling facilities to deserialize pickled object files to memory. This function also facilitates the device to load the data into (see Saving & Loading Model Across Devices).
torch.nn.Module.load_state_dict: Loads a model’s parameter dictionary using a deserialized state_dict. For more information on state_dict, see What is a state_dict?.
'''

# . A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5) # in 3 ; out 6 ; 5x5
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
opt = opt.SGD(net.parameters(), lr=0.001, momentum=0.9)
print("-"*100)
print("model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print("-"*100)
print("optimizer's state_dict:")
for var_name in opt.state_dict():
    print(var_name, "\t", opt.state_dict()[var_name])


##### NOW SAVE AND LOAD (parameters / state_dict)
print("-"*100)
PATH = './net001.pt'
torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))
net.eval()
print(net)
# Remember that you must call model.eval() to set dropout
# and batch normalization layers to evaluation mode before
# running inference. Failing to do this will yield
# inconsistent inference results.

##### NOW SAVE AND LOAD (entire model)
print("-"*100)
torch.save(net, PATH)
net = torch.load(PATH)
net.eval()
print(net)
# The disadvantage of this approach is that the serialized
# data is bound to the specific classes and the exact directory
# structure used when the model is saved


##### NOW SAVE AND LOAD
##  a General Checkpoint for Inference and/or Resuming Training
'''
# save
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)

# load
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
'''

# When saving a general checkpoint, to be used for either
# inference or resuming training, you must save more than just
# the model’s state_dict. It is important to also save the optimizer’s
# state_dict, as this contains buffers and parameters that are
# updated as the model trains.


### save on gpu . load on cpu
# # Save:
# torch.save(model.state_dict(), PATH)

# # Load:
# device = torch.device('cpu')
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH, map_location=device))

'''
Save on GPU, Load on GPU
Save:

torch.save(model.state_dict(), PATH)
Load:

device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
'''

'''
Save on CPU, Load on GPU
Save:

torch.save(model.state_dict(), PATH)
Load:

device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = i
'''