import torch
import torchmetrics
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy

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
                                depths=[3, 3, 9],
                                dims=[96, 192, 384],
                                drop_path_rate=0.,
                                layer_scale_init_value=1e-6,
                                head_init_scale=1.,
                                stem_stride=1)
        else:
            raise Exception("Model could not be found")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = self.loss(x, y)

        self.log("train loss", loss)
        self.train_metric(x, y)

        return loss

    def training_epoch_end(self, outs):
        self.log("train acc", self.train_metric, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)

        self.valid_metric(x, y)

    def validation_epoch_end(self, outs):
        self.log("valid acc", self.valid_metric, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)

        self.valid_metric(x, y)

    def test_epoch_end(self, outs):
        self.log("test acc", self.valid_metric, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}