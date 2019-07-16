from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

from data import image_loader, imshow
from model import run_style_transfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

style_img = image_loader("./img/picasso.jpg")
content_img = image_loader("./img/dancing.jpg")
input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
plt.ion()
plt.figure()
imshow(input_img, title='Input Image')

# Finally, we can run the algorithm.
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img,content_weight=0.1)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()







input("Press <ENTER> to continue ... ")

