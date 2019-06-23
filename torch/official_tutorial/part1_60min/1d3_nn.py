import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # ??
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x)) # ??
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:] # all dim except the batch dim.
                            # in torch. 1st dim is batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

params = list(net.parameters()) # net.parameters() itself is a tensor
print(len(params))
print(params[0].size()) # conv1's .weight

print("-"*333)
for i in range(len(params)):
    print(params[i].size())

print("-----------------------")
input = torch.randn(1,1,32,32) # torch convention (batch, chan, h, w)
out = net(input)
print(out)

# only supports mini-batches. The entire torch.nn package
# only supports inputs that are a mini-batch of samples,
# and not a single sample.
# If you have a single sample, just use input.unsqueeze(0)
# to add a fake batch dimension.
# or can i just do: x = torch.tensor([x]) instead of tensor(x)



net.zero_grad()
out.backward(torch.randn(1,10)) # ?? wtf is the torch.randn do here?

print("-----------------------")
print("loss function")
print("-----------------------")

output = net(input)
target = torch.randn(10)
target = target.view(1,-1) # make it same shape as output ??
# criterion = nn.MSELoss()
# loss = criterion(output, target)
loss = nn.MSELoss()(output, target)
print(loss, "\n")

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
print("\n")

print("-----------------------")
print("Backprop")
print("-----------------------")

net.zero_grad()
print('conv1.bias.grad BEFORE backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad AFTER backward')
print(net.conv1.bias.grad)

LR = 0.01 # learning rate
for f in net.parameters():
    f.data.sub_(f.grad.data * LR)

# or alternatively

import torch.optim as opt

opt = opt.SGD(net.parameters(), lr=LR)
opt.zero_grad()
output = net(input)
loss = nn.MSELoss()(output, target)
loss.backward()
opt.step()


# Question left behind: what is tensor.view do
# x = x.view(-1, self.num_flat_features(x)) # ??
# target = target.view(1,-1)
# https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch

# The view function is meant to reshape the tensor.
# a = torch.range(1, 16)
# a = a.view(4, 4)

# Drawing a similarity between numpy and pytorch, view is similar to numpy's reshape function.
