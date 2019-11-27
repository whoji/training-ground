# based on this : https://discuss.pytorch.org/t/how-to-implement-two-networks-with-shared-weights-and-with-separate-batch-normalizations/6902

import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(...)
        self.conv2 = nn.Conv2d(...)
        self.conv3 = nn.Conv2d(...)
        self.bn1 = nn.BatchNorm2d(...)
        self.bn2 = nn.BatchNorm2d(...)

    def forward(x, data_type):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if data_type == 'type1':
            x = self.bn1(x)
        elif data_type == 'type2':
            x = self.bn2(x)

        return x


optimizer.zero_grad()
output = model(data, data_type)
loss = criterion(output, target)
loss.backward()
optimizer.step()
