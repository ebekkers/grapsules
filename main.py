import torch
import argparse
import os
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='weight decay')
    parser.add_argument('--print', type=int, default=100,
                        help='print interval')
    parser.add_argument('--log', type=bool, default=False,
                        help='logging flag')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers in dataloader')
    parser.add_argument('--save_dir', type=str, default="saved models",
                        help='Directory in which to save models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Data parameters
    parser.add_argument('--dataset', type=str, default="CUB",
                        help='Data set')
    parser.add_argument('--root', type=str, default="data",
                        help='Data set location')
    parser.add_argument('--download', type=bool, default=True,
                        help='Download flag')

    # QM9 parameters
    parser.add_argument('--target', type=int, default=1,
                        help='Target see torch_geometric QM9')
    parser.add_argument('--radius', type=float, default=0.,
                        help='Radius (Angstrom) between which atoms to add links.')

    # Model parameters
    parser.add_argument('--model', type=str, default="convnext",
                        help='Model name (poni_0, poni_m, poni_12)')
    parser.add_argument('--hidden_features', type=int, default=128,
                        help='max degree of hidden rep')
    parser.add_argument('--layers', type=int, default=7,
                        help='Number of message passing layers')
    parser.add_argument('--norm', type=str, default="batch",
                        help='Normalisation type [None, batch]')
    parser.add_argument('--droprate', type=float, default=0.,
                        help='Dropout in the conv blocks.')
    parser.add_argument('--pool', type=str, default="avg",
                        help='Pooling type type [avg, sum]')
    parser.add_argument('--conv_depth', type=int, default=1,
                        help='Layers in convolution operator (1= linear conv, >1 is non-lin conv)')
    parser.add_argument('--cond_method', type=str, default="weak",
                        help='How to condition the message layers [weak, strong, pure]')
    parser.add_argument('--cond_depth', type=int, default=1,
                        help='How many layers in the message net to condition')
    parser.add_argument('--use_x_i', type=int, default=False,
                        help='Whether or not to (weakly/cat) condition the convolution on central node feature')
    parser.add_argument('--embedding', type=str, default="mlp",
                        help='The method for embedding the edge vectors [identity, mlp]')
    parser.add_argument('--sigma', type=float, default=0.2,
                        help='For the RFF feature embedding')

    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=0, type=int,
                        help='number of gpus to use (assumes all are on one node)')

    args = parser.parse_args()

    # Devices
    if args.gpus == -1:
        args.gpus = torch.cuda.device_count()
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()
    if args.gpus > 1:
        args.batch_size = int(args.batch_size / args.gpus)

    # Load the dataset and set the dataset specific settings
    if args.dataset == "CIFAR10":
        from torchvision.datasets import CIFAR10
        from modeling.pl_modules.cifar10 import CIFAR10Model as Model

        transform_train = transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(size=[32, 32], padding=4),
                    transforms.AutoAugment(torchvision.transforms.autoaugment.AutoAugmentPolicy("cifar10")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

        train_dataset = CIFAR10(args.root, train=True, transform=transform_train, download=args.download)
        test_dataset = CIFAR10(args.root, train=False, transform=transform_test, download=args.download)
        datasets = {'train': train_dataset, 'valid': test_dataset, 'test': test_dataset}
    else:
        raise Exception("Dataset could not be found")

    # Make the dataloaders
    dataloaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == 'train'), num_workers=args.num_workers)
        for split, dataset in datasets.items()}

    # The model
    model = Model(args)

    # logging
    if args.log:
        logger = pl.loggers.WandbLogger(project="GRAPSULES-" + args.dataset, name=args.model, config=args)
    else:
        logger = None

    pl.seed_everything(args.seed, workers=True)

    # Do the training and testing
    callbacks = []
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    trainer = pl.Trainer(gpus=args.gpus, logger=logger, max_epochs=args.epochs, callbacks=callbacks,
                         gradient_clip_val=0.5)
    trainer.fit(model, dataloaders['train'], dataloaders['valid'])
    trainer.test(model, dataloaders['test'])

