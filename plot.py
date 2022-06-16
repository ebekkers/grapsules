import torch
import argparse
import os
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import io
import PIL

import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'

from torchvision.datasets import CIFAR10
transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])


test_dataset = CIFAR10("data", train=False, transform=transform_test, download=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

imstack, labels = next(iter(test_loader))
[B,C,X,Y] = imstack.shape
heatmaps = torch.softmax(torch.randn([B,X*Y]),-1).reshape(B,X,Y)

def heatmap_grid(imstack, heatmaps):
    # imstack.shape = [B, 3, X, Y]
    # heatmaps.shape = [B, X, Y]
    imstack2show = (imstack - imstack.min())/(imstack.max() - imstack.min())
    heatmaps_rescaled = heatmaps / torch.amax(heatmaps,(-1,-2))[:,None,None]
    imstack2show[:, 0, :, :] = imstack2show[:, 0, :, :] * (1 - 0.5 * heatmaps_rescaled) + 0.5 * heatmaps_rescaled
    image_grid = torchvision.utils.make_grid(imstack2show, 4, 0).permute([1,2,0])
    plt.imshow(image_grid)
    plt.show()

heatmap_grid(imstack, heatmaps)

imstack2show = (imstack - imstack.min())/(imstack.max() - imstack.min())
plt.imshow(imstack2show[0].permute([1,2,0]))
pos = torch.tensor([[0,0],[1,0],[10,10]]).type_as(imstack2show)
plt.scatter(pos[:,0], pos[:,1], s=100, c=np.linspace(0,1,len(pos)), alpha=0.8)
plt.show()



def get_coords(h, w):
    # return a coordinate grid over [0, 1] interval with h (heigh) and w (width) sample density
    range_x = torch.tensor(np.linspace(0, h - 1, h))
    range_y = torch.tensor(np.linspace(0, w - 1, w))

    xx, yy = torch.meshgrid((range_x, range_y), indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

pos_grid = get_coords(32,32)
pos = 15 + 4*torch.rand(B, 24, 2)

def sample_gaussians(grid, mus, sigma):
    rel_pos = grid[None, :, None, :] - mus[:,None, :, :]
    sampled = (1/(2*torch.pi*sigma**2)) * torch.exp(-0.5 * torch.sum(rel_pos**2,-1) / (sigma**2))  # [B, XY, H]
    return sampled

sampled = sample_gaussians(pos_grid, pos,3)
print(torch.sum(sampled,dim=-2))
plt.figure()
plt.imshow(sampled[0,:,0].reshape([X, Y]))
plt.show()
plt.figure()
plt.imshow(sampled[0,:,1].reshape([X, Y]))
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc



poss = 15 + 10*torch.rand(3, B, 3, 2)
imnr = 0
plt.figure()
fig = plt.imshow(imstack2show[imnr].permute([1, 2, 0]).cpu().detach().numpy())
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
for it in range(len(poss)):
    pos = poss[it]
    plt.scatter(pos[imnr, :, 1].cpu().detach().numpy(), pos[imnr, :, 0].cpu().detach().numpy(), s=50 + it * 50,
                c=np.linspace(0, 1, pos.shape[1]), alpha=0.75, cmap='jet')
import matplotlib.pylab as pl
# plt.plot(poss[:, imnr, :, 1].cpu().detach().numpy(), poss[:, imnr, :, 0].cpu().detach().numpy(), alpha=0.75, colors=plt.cm.jet(np.linspace(0,1,poss.shape[2])))
multiline(poss[:, imnr,:,1].cpu().detach().numpy().transpose(), poss[:, imnr,:,0].cpu().detach().numpy().transpose(), c=np.linspace(0, 1, pos.shape[1]), cmap='jet', lw=2)
plt.show()