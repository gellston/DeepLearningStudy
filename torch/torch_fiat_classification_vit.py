import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from PIL import Image





img = Image.open('C://Github//DeepLearningStudy//test_image//aigul.jpg')
fig = plt.figure()
plt.imshow(img)


transform = Compose([Resize((224, 224)), ToTensor()])


x_input = transform(img)
x_input = x_input.unsqueeze(0)
print('shape = ', x_input.shape)


patch_size = 16
patches = rearrange(x_input, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
print('patch test shape = ', patches.shape)




class PatchEmbedding(torch.nn.Module):
    def __init__(self,
                 in_channels=3,
                 patch_size=16):

        self.patch_size = patch_size
        super().__init__()
        embedding_size = patch_size * patch_size * in_channels
        self.projection = torch.nn.Sequential(
            #Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            #torch.nn.Linear(patch_size * patch_size * in_channels, embedding_size)
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=embedding_size,
                            kernel_size=patch_size,
                            stride=patch_size,
                            padding=0),
            Rearrange('b c (h) (w) -> b (h w) c')
        )

    def forward(self, x):
        x = self.projection(x)
        return x




torch_output = PatchEmbedding()(x_input)

print('torch_output shape = ', torch_output.shape)


plt.waitforbuttonpress()