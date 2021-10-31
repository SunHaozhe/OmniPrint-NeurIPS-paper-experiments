
import os
import sys
import glob
import datetime
sys.path.append(os.path.join(os.pardir, "omniprint"))
sys.path.append(os.path.join(os.pardir, "omniprint", "dataloader"))

import argparse
import collections
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim
from torchvision import transforms
from sklearn.metrics import r2_score

import wandb

from dataloader import MultilingualDataset

from utils import *



class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(6, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.sequential(x)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MultilingualDatasetRegressionIntermediate(MultilingualDataset):
    def __init__(self, data_path, regression_label, image_extension=".png"):
        self.regression_label = regression_label
        super().__init__(data_path, transform=None, target_transform=None, 
            label="unicode_code_point", image_extension=image_extension) 
        
    def consume_raw_labels_csv(self):
        dfs = dict()

        search_pattern = os.path.join(self.data_path, "*", "label", "raw_labels.csv")
        for path in sorted(glob.glob(search_pattern)):
            alphabet_name = path.split(os.sep)[-3]
            df = pd.read_csv(path, sep="\t", encoding="utf-8")
            dfs[alphabet_name] = df.loc[:, ["image_name", self.label, self.regression_label]]
        return dfs

    def construct_items(self):
        """
        Builds self.items, self.idx2raw_label
        
        item: tuple
            (path, raw_label, regression_label, label/target)

        idx2raw_label: dict 
            unique ID of raw_label ==> the value of raw_label
        """
        # dict, raw labels for each alphabet
        dfs = self.consume_raw_labels_csv()

        self.items = []

        search_pattern = os.path.join(self.data_path, "*", "data", "*" + self.image_extension)
        for path in sorted(glob.glob(search_pattern)):
            file_name = os.path.basename(path)
            alphabet_name = path.split(os.sep)[-3]
            df = dfs[alphabet_name]
            raw_label = df.loc[df["image_name"] == file_name, self.label].iloc[0]
            regression_label = df.loc[df["image_name"] == file_name, self.regression_label].iloc[0]
            self.items.append([path, raw_label, regression_label])

        idx = dict() 
        for item in self.items:
            if item[1] not in idx:
                idx[item[1]] = len(idx)
            item.append(idx[item[1]])

        self.idx2raw_label = {v: k for k, v in idx.items()} 

    def __getitem__(self, index):
        img_path = self.items[index][0]
        regression_label = self.items[index][2]
        target = self.items[index][3]
        return (img_path, regression_label), target


class ImagePathDataset(torch.utils.data.Dataset):
    """
    tuples: list of lists
        each element is a tuple of two elements
        the first is the image path, the second is the regression label
    """
    def __init__(self, tuples, transform=None, image_mode="L"):
        super().__init__()
        self.transform = transform
        self.image_mode = image_mode

        self.items = []
        for sublist in tuples:
            for tuple_ in sublist:
                self.items.append(tuple_)

    def __len__(self):
        return len(self.items)


    def __getitem__(self, index):
        img_path, label = self.items[index]
        img = Image.open(img_path).convert(self.image_mode)
        if self.transform is not None:
            img = self.transform(img)
        return img, label



def get_dataloaders(data_path, train_classes, batch_size, image_size=32):
    """
    data_path: str
        e.g. ../out/multilingual/20210502_172424_943937
        Under data_path, there should be several directories, each of which 
        corresponds to one alphabet. In each alphabet directory, one should have 
        two subdirectories: data and label. The subdirectory data contains the images 
        e.g. arabic_0.png. The subdirectory label contains a single file raw_labels.csv.
    """
    print("get_dataloaders starts...")
    if train_classes is not None:
        assert train_classes <= 900, "train_classes must be smaller than 900, but got {}".format(train_classes)
    else:
        train_classes = 900
    print("Creating MultilingualDatasetRegressionIntermediate instance...")
    initial_dataset = MultilingualDatasetRegressionIntermediate(data_path=data_path, 
                                                                regression_label=args.regression_label)
    print("MultilingualDatasetRegressionIntermediate instance ready.")

    # transform OmniglotLikeDataset to a np.array (self.dataset)
    label2imgs = collections.defaultdict(list)
    for img, target in initial_dataset:
        label2imgs[target].append(img)
    initial_dataset = []
    for target, imgs in label2imgs.items():
        initial_dataset.append(imgs)
    
    # initial_dataset is a list of lists (1409 x 20), 
    # each element is a tuple of two elements, 
    # the first is the image path, the second is the regression label

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        "test": transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    }

    print("Building ImagePathDataset for train, val and test...")

    # use the (almost) same split as the few-shot learning use case
    # 900/149/360 where the first number can be reduced
    train_dataset = ImagePathDataset(initial_dataset[:train_classes], transform=data_transforms["train"])
    val_dataset = ImagePathDataset(initial_dataset[train_classes:1049], transform=data_transforms["test"])
    test_dataset = ImagePathDataset(initial_dataset[1049:], transform=data_transforms["test"])

    print("ImagePathDataset for train, val and test ready.")

    print("Building torch.utils.data.DataLoader for train, val and test...")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   

    print("torch.utils.data.DataLoader for train, val and test ready.")

    return train_dataloader, val_dataloader, test_dataloader


def r2_from_logits_y(logits, y):
    logits = logits.detach().cpu().numpy().ravel()
    y = y.detach().cpu().numpy().ravel()
    return r2_score(logits, y)


def train(train_dataloader, model, device, optim, epoch, wandb):
    t0 = time.time()

    train_loss = 0
    train_r2 = 0
    model.train()
    for batch_idx, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        optim.zero_grad()

        logits = model(X).float().view(-1)
        y = y.float().view(-1)
        loss = F.mse_loss(logits, y)

        loss.backward()
        optim.step()

        train_loss += loss.item()

        train_r2 += r2_from_logits_y(logits, y)

    train_loss /= len(train_dataloader)
    train_r2 /= len(train_dataloader)
    
    t1 = time.time() - t0
    print("Epoch {} | Train loss {:.2f} | Train R2 {:.2f} | Time {:.1f} seconds.".format(
            epoch+1, train_loss, train_r2, t1))
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_r2": train_r2, "train_epoch_time": t1})

def val(val_dataloader, model, device, epoch, best_model, scheduler, wandb):
    t0 = time.time()
    
    val_loss = 0
    val_r2 = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_dataloader):
            X, y = X.to(device), y.to(device)

            logits = model(X).float().view(-1)
            y = y.float().view(-1)
            loss = F.mse_loss(logits, y)

            val_loss += loss.item()

            val_r2 += r2_from_logits_y(logits, y)
    
    val_loss /= len(val_dataloader)
    val_r2 /= len(val_dataloader)

    # save model checkpoint
    if val_loss < best_model[0]:
        best_model[0] = val_loss
        torch.save(model.state_dict(), best_model[1])

    t1 = time.time() - t0
    print("Epoch {} | Val loss {:.2f} | Val R2 {:.2f} | Time {:.1f} seconds.".format(
            epoch+1, val_loss, val_r2, t1))
    wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_r2": val_r2, "val_epoch_time": t1})

    # scheduling learning rates
    scheduler.step(val_loss)

def test(test_dataloader, device, best_model, wandb):
    t0 = time.time()

    model = Backbone().to(device)
    model.load_state_dict(torch.load(best_model[1], map_location=device))
    
    test_loss = 0
    test_r2 = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)

            logits = model(X).float().view(-1)
            y = y.float().view(-1)
            loss = F.mse_loss(logits, y)

            test_loss += loss.item()

            test_r2 += r2_from_logits_y(logits, y)
    
    test_loss /= len(test_dataloader)
    test_r2 /= len(test_dataloader)

    t1 = time.time() - t0
    print("Test loss {:.2f} | Test R2 {:.2f} | Time {:.1f} seconds.".format(test_loss, test_r2, t1))
    wandb.log({"test_loss": test_loss, "test_r2": test_r2, "test_epoch_time": t1}) 


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='which dataset to test', 
                            default='regression_large_dataset')
    argparser.add_argument('--image_size', type=int, help='size of input images of neural networks', default=32)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--lr', type=float, help='learning rate of the optimizer', default=1e-3)
    argparser.add_argument('--momentum', type=float, default=0.9)
    argparser.add_argument('--weight_decay', type=float, default=1e-5)
    argparser.add_argument('--scheduler_patience', type=int, default=5)
    argparser.add_argument('--scheduler_factor', type=float, default=0.1)
    argparser.add_argument('--epochs', type=int, help='how many epochs in total', default=30)
    argparser.add_argument('--random_seed', type=int, help='random seed', default=68)
    argparser.add_argument('--regression_label', type=str, default="shear_x", choices=["shear_x", "rotation"])
    argparser.add_argument('--train_classes', type=int, help='how many classes to use for training', default=None)
    args = argparser.parse_args()

    t0_overall = time.time()

    # random seed
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU for PyTorch")
    else:
        device = torch.device('cpu')
        print("Using CPU for PyTorch")


    # neural network
    model = Backbone().to(device)

    print("Model created.")

    # optimizer and scheduler
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                                                           "min", 
                                                           verbose=True, 
                                                           patience=args.scheduler_patience, 
                                                           factor=args.scheduler_factor)
    print("Optimizer and scheduler ready.")

    # data loaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_path=args.dataset, 
                                                                        train_classes=args.train_classes, 
                                                                        batch_size=args.batch_size, 
                                                                        image_size=args.image_size)

    
    # wandb
    project_name = "OmniPrint-regression-{}-{}".format(os.path.basename(args.dataset), args.regression_label)
    group_name = "train_classes-{}".format(args.train_classes)
    wandb_dir = "wandb_logs"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    wandb.init(config=args, project=project_name, group=group_name, dir=wandb_dir)
    
    env_info = get_torch_gpu_environment()
    for k, v in env_info.items():
        wandb.run.summary[k] = v
    wandb.run.summary["trainable_parameters_count"] = count_trainable_parameters(model)


    # checkpoint setting
    checkpoints_dir = "model_checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    timestamp = str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))
    best_model = [np.inf, os.path.join(checkpoints_dir, 
        "best_model_{}_{}_{}.pth".format(project_name, group_name, timestamp))] # (score, path)

    print("Training loop starts...")

    for epoch in range(args.epochs):
        train(train_dataloader, model, device, optim, epoch, wandb)
        val(val_dataloader, model, device, epoch, best_model, scheduler, wandb)
    test(test_dataloader, device, best_model, wandb)


    t_overall = time.time() - t0_overall 
    print("Done in {:.2f} s.".format(t_overall))
    wandb.run.summary["overall_computation_time"] = t_overall




