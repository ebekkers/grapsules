import torch
import torchmetrics
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy
import wandb



# For plot
import numpy as np
import io
import PIL
import matplotlib.pyplot as plt
import wandb
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


class CIFAR10Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs

        self.loss = cross_entropy
        self.train_metric = torchmetrics.Accuracy()
        self.valid_metric = torchmetrics.Accuracy()

        if args.model == "convnext":
            from ..base_models.convnext import ConvNeXt
            self.net = ConvNeXt(in_chans=3,
                                num_classes=10,
                                depths=[6, 9],
                                dims=[384, 128],
                                drop_path_rate=0.2,
                                layer_scale_init_value=1e-6,
                                head_init_scale=1.,
                                stem_stride=1)
        elif args.model == "grapsulenet":
            from ..base_models.grapsulenet import GrapsuleNet
            self.net = GrapsuleNet(in_chans=3,
                                   num_classes=10,
                                   block_depth=6, dim=256, num_heads=16, num_it=3,
                                   drop_path_rate=0., layer_scale_init_value=0, stem_stride=1)
        else:
            raise Exception("Model could not be found")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, preds, _ = self.forward(x)
        # loss = self.loss(x, y)
        # loss = 0.
        # for pred in preds:
        #     loss += self.loss(pred, y)
        pred = torch.stack(preds, dim=0).mean(dim=0)
        loss = self.loss(pred, y)


        self.log("train loss", loss)
        # self.logger.experiment.log({"train loss": loss, "pos": self.net.im_to_log,  "heat map": self.net.im_to_log2, "penalty": self.net.penalty})
        self.train_metric(x, y)

        return loss

    def training_epoch_end(self, outs):
        self.log("train acc", self.train_metric, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        imstack2show = (x - x.min()) / (x.max() - x.min())

        x, _, poss = self.forward(x)

        self.valid_metric(x, y)

        # Plot log
        # imnr = 0
        # imlist = {}
        # for it in range(len(poss)):
        #     pos = poss[it]
        #     buf = io.BytesIO()
        #     plt.figure()
        #     fig = plt.imshow(imstack2show[imnr].permute([1, 2, 0]).cpu().detach().numpy())
        #     fig.axes.get_xaxis().set_visible(False)
        #     fig.axes.get_yaxis().set_visible(False)
        #     plt.scatter(pos[imnr, :, 1].cpu().detach().numpy(), pos[imnr, :, 0].cpu().detach().numpy(), s=100,
        #                 c=np.linspace(0, 1, pos.shape[1]), alpha=0.8, cmap='jet')
        #     plt.savefig(buf)
        #     buf.seek(0)
        #     img = PIL.Image.open(buf)
        #     plt.close()
        #     imlist["pos_"+str(it)] = wandb.Image(img)
        # self.logger.experiment.log(imlist)
        imnr = 0
        buf = io.BytesIO()
        plt.figure()
        fig = plt.imshow(imstack2show[imnr].permute([1, 2, 0]).cpu().detach().numpy())
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # Lines
        xs = torch.stack(poss, dim=-1)[imnr,:,1,:].cpu().detach().numpy()
        ys = torch.stack(poss, dim=-1)[imnr,:,0,:].cpu().detach().numpy()
        multiline(xs, ys, c=np.linspace(0, 1, xs.shape[0]), cmap='jet', lw=2)
        for it in range(len(poss)):
            pos = poss[it]
            plt.scatter(pos[imnr, :, 1].cpu().detach().numpy(), pos[imnr, :, 0].cpu().detach().numpy(), s=50+it*50,
                        c=np.linspace(0, 1, pos.shape[1]), alpha=0.5, cmap='jet')
        plt.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)
        plt.close()
        self.logger.experiment.log({"pos":wandb.Image(img)})


    def validation_epoch_end(self, outs):
        self.log("valid acc", self.valid_metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, _, _ = self.forward(x)

        self.valid_metric(x, y)

    def test_epoch_end(self, outs):
        self.log("test acc", self.valid_metric, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}