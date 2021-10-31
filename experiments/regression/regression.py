
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
from torchvision import transforms, models
from sklearn.metrics import r2_score

import wandb

from dataloader import MultilingualDataset

from utils import *



class Backbone(nn.Module):
    def __init__(self, image_mode):
        super().__init__()
        if image_mode == "L":
            in_channels_ = 1
        elif image_mode == "RGB":
            in_channels_ = 3

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_, out_channels=64, kernel_size=3),
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
        Builds self.items
        
        item: tuple
            (path, raw_label, regression_label, label/target)

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


    def __getitem__(self, index):
        img_path = self.items[index][0]
        regression_label = self.items[index][2]
        target = self.items[index][3]
        return (img_path, regression_label), target


class ImagePathDataset(torch.utils.data.Dataset):
    """
    tuples: list of tuple
        each tuple consists of two elements
        the first is the image path, the second is the regression label
    """
    def __init__(self, tuples, transform=None, image_mode="L"):
        super().__init__()
        self.transform = transform
        self.image_mode = image_mode
        self.items = tuples

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path, label = self.items[index]
        img = Image.open(img_path).convert(self.image_mode)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def indices_to_list_of_tuples(initial_dataset, indices):
    # tuples: list of tuple
    tuples = []
    for idx in indices:
        (img_path, regression_label), target = initial_dataset[idx]
        tuples.append((img_path, regression_label))
    return tuples

def get_dataloaders(data_path, train_instances, val_instances, test_instances, batch_size, 
    image_size=32, wandb=None, image_mode="L", backbone="small"):
    """
    data_path: str
        e.g. ../out/multilingual/20210502_172424_943937
        Under data_path, there should be several directories, each of which 
        corresponds to one alphabet. In each alphabet directory, one should have 
        two subdirectories: data and label. The subdirectory data contains the images 
        e.g. arabic_0.png. The subdirectory label contains a single file raw_labels.csv.
    """
    

    t_intermediate_dataset_0 = time.time()

    print("Creating MultilingualDatasetRegressionIntermediate instance...")

    initial_dataset = MultilingualDatasetRegressionIntermediate(data_path=data_path, 
                                                                regression_label=args.regression_label)

    t_intermediate_dataset_1 = time.time() - t_intermediate_dataset_0

    print("MultilingualDatasetRegressionIntermediate instance ready in {:.1f} s.".format(t_intermediate_dataset_1))

    if wandb is not None:
        wandb.run.summary["MultilingualDatasetRegressionIntermediate_time"] = t_intermediate_dataset_1
    
    total_nb_instances = len(initial_dataset)
    assert train_instances + val_instances + test_instances <= total_nb_instances


    # determine the train/val/test split
    t0_ = time.time()
    train_indices = np.random.choice(total_nb_instances, size=train_instances, replace=False)
    remaining_indices = [xx for xx in np.arange(total_nb_instances) if xx not in train_indices]
    val_indices = np.random.choice(remaining_indices, size=val_instances, replace=False)
    remaining_indices = [xx for xx in remaining_indices if xx not in val_indices]
    test_indices = np.random.choice(remaining_indices, size=test_instances, replace=False)
    t1_ = time.time() - t0_
    print("Indices split into train/val/test in {:.1f} s.".format(t1_))
    if wandb is not None:
        wandb.run.summary["split_idx_time"] = t1_

    t0_ = time.time()
    train_tuples = indices_to_list_of_tuples(initial_dataset, train_indices)
    val_tuples = indices_to_list_of_tuples(initial_dataset, val_indices)
    test_tuples = indices_to_list_of_tuples(initial_dataset, test_indices)
    t1_ = time.time() - t0_
    print("Tuples split into train/val/test in {:.1f} s.".format(t1_))
    if wandb is not None:
        wandb.run.summary["split_tuple_time"] = t1_


    if image_mode == "L":
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
    elif image_mode == "RGB":
        data_transforms = {
            "train": transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            "test": transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        }

    if backbone == "resnet18":
        resnet18_input_size = 224
        data_transforms = {
            "train": transforms.Compose([
                transforms.Resize(resnet18_input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "test": transforms.Compose([
                transforms.Resize(resnet18_input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
    print("Building ImagePathDataset for train/val/test...")

    t0_ = time.time()
    train_dataset = ImagePathDataset(train_tuples, transform=data_transforms["train"], image_mode=image_mode)
    val_dataset = ImagePathDataset(val_tuples, transform=data_transforms["test"], image_mode=image_mode)
    test_dataset = ImagePathDataset(test_tuples, transform=data_transforms["test"], image_mode=image_mode)
    t1_ = time.time() - t0_
    print("ImagePathDataset for train/val/test ready in {:.1f} s.".format(t1_))
    if wandb is not None:
        wandb.run.summary["ImagePathDataset_time"] = t1_

    

    print("Building torch.utils.data.DataLoader for train/val/test...")
    t0_ = time.time()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   
    t1_ = time.time() - t0_
    print("torch.utils.data.DataLoader for train/val/test ready in {:.1f} s.".format(t1_))
    if wandb is not None:
        wandb.run.summary["torch_DataLoader_time"] = t1_
    
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

def test(test_dataloader, device, best_model, wandb, args):
    t0 = time.time()

    model = get_backbone(args, device)
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
    wandb.run.summary["test_loss"] = test_loss
    wandb.run.summary["test_r2"] = test_r2
    wandb.run.summary["test_epoch_time"] = t1


def get_backbone(args, device):
    if args.backbone == "resnet18":
        assert args.image_mode == "RGB"
        # in_features == 512, 
        ## last layer: (512 + 1) * 1 = 513 trainable parameters
        ## layer4.1.bn2: 1024 trainable parameters
        ## layer4.1.conv2: 2359296 trainable parameters
        ### In total, 2 360 833 (2.36 million) trainable parameters
        model = models.resnet18(pretrained=True)
        for name, param in model.named_parameters():
            if name.startswith("fc"):
                param.requires_grad = True
            elif name.startswith("layer4.1.bn2"):
                param.requires_grad = True
            elif name.startswith("layer4.1.conv2"):
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        fc_in_features = model.fc.in_features
        model.fc = nn.Linear(fc_in_features, 1)
        model.to(device)
    elif args.backbone == "small":
        # 74945 trainable parameters
        model = Backbone(args.image_mode).to(device)
    else:
        raise NotImplementedError
    return model


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='which dataset to test', 
                            default='regression_large_dataset')
    argparser.add_argument('--image_size', type=int, help='size of input images of neural networks', default=32)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--lr', type=float, help='learning rate of the optimizer', default=1e-3)
    argparser.add_argument('--momentum', type=float, default=0.9)
    argparser.add_argument('--weight_decay', type=float, default=1e-4)
    argparser.add_argument('--scheduler_patience', type=int, default=5)
    argparser.add_argument('--scheduler_factor', type=float, default=0.1)
    argparser.add_argument('--epochs', type=int, help='how many epochs in total', default=30)
    argparser.add_argument('--random_seed', type=int, help='random seed', default=68)
    argparser.add_argument('--regression_label', type=str, default="shear_x", choices=["shear_x", "rotation"])
    argparser.add_argument('--train_instances', type=int, help='how many images to use for training', required=True)
    argparser.add_argument('--val_instances', type=int, help='how many images to use for validation', required=True)
    argparser.add_argument('--test_instances', type=int, help='how many images to use for test', required=True)
    argparser.add_argument('--image_mode', type=str, default="RGB", choices=["L", "RGB"])
    argparser.add_argument('--backbone', type=str, default="resnet18", choices=["resnet18", "small"])
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
    model = get_backbone(args, device)

    print("Model created.")
    
    # wandb
    project_name = "OmniPrint-regressionMixV2-{}-{}".format(os.path.basename(args.dataset), args.regression_label)
    group_name = "{}-{}".format(args.backbone, args.train_instances)
    wandb_dir = "wandb_logs"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    wandb.init(config=args, project=project_name, group=group_name, dir=wandb_dir)
    
    env_info = get_torch_gpu_environment()
    for k, v in env_info.items():
        wandb.run.summary[k] = v
    wandb.run.summary["trainable_parameters_count"] = count_trainable_parameters(model)


    

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
                                                                        train_instances=args.train_instances, 
                                                                        val_instances=args.val_instances, 
                                                                        test_instances=args.test_instances, 
                                                                        batch_size=args.batch_size, 
                                                                        image_size=args.image_size,
                                                                        wandb=wandb,
                                                                        image_mode=args.image_mode,
                                                                        backbone=args.backbone)

    


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
    test(test_dataloader, device, best_model, wandb, args)

    
    t_overall = time.time() - t0_overall 
    print("Done in {:.2f} s.".format(t_overall))
    wandb.run.summary["overall_computation_time"] = t_overall




