

import argparse
import time
import typing

import pandas as pd
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
import datetime
sys.path.append(os.path.join(os.pardir, "omniprint"))
sys.path.append(os.path.join(os.pardir, "omniprint", "dataloader"))

from dataloader import OmniglotLikeDataloader

import wandb

from utils import *

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='which dataset to test', 
                            default='../omniprint/omniglot_like_datasets/meta1')
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_support', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_query', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--batch_size', type=int, help="""meta batch size, namely task num, 
        each epoch contains 1 batch""", default=1)
    argparser.add_argument('--batch_size_non_episodic', type=int, help="""batch size for the 
        non-episodic dataloader""", default=32)
    argparser.add_argument('--image_size', type=int, help='size of input images of neural networks', default=32)
    argparser.add_argument('--finetune_inner_steps', type=int, 
        help='number of gradient steps for each test episodes', default=20)
    argparser.add_argument('--epochs', type=int, help='how many epochs in total', default=600)
    argparser.add_argument('--lr', type=float, help='learning rate of the optimizers', default=5e-5)
    argparser.add_argument('--weight_decay', type=float, help='weight decay of the meta-train optimizer', default=1e-3)
    argparser.add_argument('--seed', type=int, help='random seed', default=68)
    argparser.add_argument('--nb_test_episodes', type=int, help="""how many episodes to use to 
        compute the meta-test accuracy""", default=1000)
    argparser.add_argument('--only_last_layer', action="store_true", help="""Whether 
        only trains the last layer or fine-tune the whole network""", default=False)
    argparser.add_argument('--train_episodes', type=int, default=57600, help="how many meta train episodes")
    args = argparser.parse_args()

    args.epochs = args.train_episodes

    t0_overall = time.time()
    
    project_name = "OmniPrint-{}-episodes_PretrainOnMtrain".format(os.path.basename(args.dataset))
    if args.only_last_layer:
        project_name += "_last"
    else:
        project_name += "_whole"
    group_name = "{}-way-{}-shot-{}".format(args.n_way, args.k_support, args.train_episodes)
    wandb_dir = "wandb_logs"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    wandb.init(config=args, project=project_name, group=group_name, dir=wandb_dir)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Set up the OmniPrint loader.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataloader = OmniglotLikeDataloader(data_path=args.dataset, 
                                        batch_size=args.batch_size,
                                        n_way=args.n_way,
                                        k_support=args.k_support,
                                        k_query=args.k_query,
                                        image_size=args.image_size,
                                        device=device,
                                        msplit="32,8,14")
    non_episodic_dataloader, n_classes = dataloader.get_non_episodic_dataloader("train")

    net = Backbone(n_classes).to(device)

    wandb.run.summary["trainable_parameters_count"] = count_trainable_parameters(net.embedding)
    
    env_info = get_torch_gpu_environment()
    for k, v in env_info.items():
        wandb.run.summary[k] = v
    
    optim = torch.optim.Adam(params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    checkpoints_dir = "model_checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    timestamp = str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))
    best_model = [-1, os.path.join(checkpoints_dir, 
        "best_model_{}_{}_{}.pth".format(project_name, group_name, timestamp))] # (score, path)

    log = []
    mval_period = min(args.epochs, 960)
    for epoch in range(args.epochs):
        mtrain(non_episodic_dataloader, net, device, optim, epoch, log)
        if epoch % mval_period == mval_period - 1:
            statedict = filter_statedict(net.state_dict())
            mval(dataloader, statedict, args.lr, args.finetune_inner_steps, 
                device, args.n_way, epoch, log, best_model, args.only_last_layer)

    mtest(dataloader, args.lr, args.finetune_inner_steps, device, args.n_way, 
        log, best_model, args.only_last_layer, nb_test_episodes=args.nb_test_episodes)

    df = pd.DataFrame(log) 
    t_overall = time.time() - t0_overall 

    #timestamp = str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))
    #df.to_csv("maml_omniglot_like_log_{}.csv".format(timestamp))

    wandb.run.summary["overall_computation_time"] = t_overall


def mtrain(non_episodic_dataloader, net, device, optim, epoch, log):
    """
    one epoch of meta-train
    """
    net.train()
    
    start_time = time.time()

    epoch_average_loss = 0
    epoch_average_acc = 0

    for batch_idx, (X, y) in enumerate(non_episodic_dataloader):
        current_batch_size = y.shape[0]

        optim.zero_grad()
        logits = net(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optim.step()

        epoch_average_loss += loss.item()
        # logits (32, 900), y (32)
        acc = (logits.argmax(dim=1) == y).sum().item() / current_batch_size
        epoch_average_acc += acc

    epoch_average_loss /= len(non_episodic_dataloader)
    epoch_average_acc /= len(non_episodic_dataloader)
    epoch_average_acc *= 100
    mtrain_epoch_time = time.time() - start_time

    print("Epoch {} Train Loss {} | Acc: {} | Time {}".format(epoch+1, 
        epoch_average_loss, epoch_average_acc, mtrain_epoch_time))

    log.append({
        "epoch": epoch+1,
        "mtrain_epoch_average_loss": epoch_average_loss,
        "mtrain_epoch_average_accuracy": epoch_average_acc,
        "mtrain_epoch_time": mtrain_epoch_time
    })
    wandb.log({"epoch": epoch+1, 
        "mtrain_epoch_average_loss": epoch_average_loss, 
        "mtrain_epoch_average_accuracy": epoch_average_acc, 
        "mtrain_epoch_time": mtrain_epoch_time})
        


def mval(dataloader, pretrained_statedict, lr, finetune_inner_steps, device, 
    n_way, epoch, log, best_model, only_last_layer):
    """
    one epoch of meta-validation
    """

    nb_batches_per_epoch_mval = 192
    
    qry_losses = []
    qry_accs = []

    start_time = time.time()

    for batch_idx in range(nb_batches_per_epoch_mval):
        x_spt, y_spt, x_qry, y_qry = dataloader.next("val")

        batch_size, _, _, _, _ = x_spt.size()
        querysz = x_qry.size(1)
        
        for i in range(batch_size):
            # one meta-validation episode/task

            net = get_new_nn(pretrained_statedict, n_way, device, only_last_layer)
            net.train()
            optim = torch.optim.Adam(params=net.parameters(), lr=lr)

            for _ in range(finetune_inner_steps):
                optim.zero_grad()
                logits_spt = net(x_spt[i])
                loss_spt = F.cross_entropy(logits_spt, y_spt[i])
                loss_spt.backward()
                optim.step()

            qry_logits = net(x_qry[i])
            qry_loss = F.cross_entropy(qry_logits, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu().item())
            qry_acc = (qry_logits.argmax(dim=1) == y_qry[i]).sum().item() / querysz
            qry_accs.append(qry_acc)

    qry_losses, qry_losses_conf = compute_mean_and_confidence_interval(qry_losses)
    qry_accs, qry_accs_conf = compute_mean_and_confidence_interval(qry_accs)
    qry_accs *= 100
    qry_accs_conf *= 100
    
    epoch_time = time.time() - start_time

    print(f'[Epoch {epoch+1:.2f}] Validation Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time {epoch_time:.2f}')

    log.append({
        'epoch': epoch + 1,
        'mval_query_loss': qry_losses,
        'mval_query_accuracy': qry_accs,
        'mval_epoch_computation_time': epoch_time,
        'mval_query_loss_confidence_interval': qry_losses_conf,
        'mval_query_accuracy_confidence_interval': qry_accs_conf
    })
    wandb.log({"mval_query_loss": qry_losses, "mval_query_accuracy": qry_accs, 
        "epoch": epoch + 1, "mval_epoch_computation_time": epoch_time,
        "mval_query_loss_confidence_interval": qry_losses_conf,
        "mval_query_accuracy_confidence_interval": qry_accs_conf})
    if qry_accs > best_model[0]:
        best_model[0] = qry_accs
        torch.save(pretrained_statedict, best_model[1])
        wandb.run.summary["best_mval_query_accuracy"] = qry_accs
        wandb.run.summary["best_mval_query_accuracy_confidence_interval"] = qry_accs_conf


def mtest(dataloader, lr, finetune_inner_steps, device, n_way, log, best_model, only_last_layer, nb_test_episodes=1000):
    """
    one epoch of meta-test
    """
    qry_accs = []
    start_time = time.time()
    while True:
        x_spt, y_spt, x_qry, y_qry = dataloader.next("test")

        batch_size, _, _, _, _ = x_spt.size()
        querysz = x_qry.size(1)
        
        for i in range(batch_size):
            # one meta-test episode/task

            net = get_new_nn(torch.load(best_model[1]), n_way, device, only_last_layer)
            net.train()
            optim = torch.optim.Adam(params=net.parameters(), lr=lr)

            for _ in range(finetune_inner_steps):
                optim.zero_grad()
                logits_spt = net(x_spt[i])
                loss_spt = F.cross_entropy(logits_spt, y_spt[i])
                loss_spt.backward()
                optim.step()

            qry_logits = net(x_qry[i])
            qry_acc = (qry_logits.argmax(dim=1) == y_qry[i]).sum().item() / querysz
            qry_accs.append(qry_acc)

        # len(qry_accs) == batch_size (it is 1-D)
        if len(qry_accs) >= nb_test_episodes:
            break

    qry_accs = qry_accs[:nb_test_episodes]
    qry_accs, qry_accs_conf = compute_mean_and_confidence_interval(qry_accs)
    qry_accs *= 100
    qry_accs_conf *= 100
    
    mtest_time = time.time() - start_time

    print(f'Test Loss: Acc: {qry_accs:.2f}')
    
    log.append({
        'mtest_query_accuracy': qry_accs,
        'mtest_query_accuracy_confidence_interval': qry_accs_conf,
        'mtest_computation_time': mtest_time
    })
    wandb.run.summary["mtest_query_accuracy"] = qry_accs
    wandb.run.summary["mtest_query_accuracy_confidence_interval"] = qry_accs_conf
    wandb.run.summary["mtest_computation_time"] = mtest_time


class Backbone(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
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
            Flatten()
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def get_new_nn(pretrained_statedict, n_way, device, only_last_layer=False):
    net = Backbone(n_way)
    statedict = net.state_dict()
    statedict.update(pretrained_statedict)
    net.load_state_dict(statedict)
    net.to(device)

    for param_name, param in net.named_parameters():
        if param_name.startswith("fc."):
            param.requires_grad = True
        else:
            if only_last_layer:
                param.requires_grad = False
            else:
                param.requires_grad = True
    return net


def filter_statedict(statedict):
    return {k: v for k, v in statedict.items() if not k.startswith("fc.")}


if __name__ == '__main__':
    main()





