"""
Adapted from https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch 

This project is licensed under the MIT License

MIT License

Copyright (c) 2018 Daniele Ciriello

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Adapted by Haozhe Sun
"""


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
    argparser.add_argument('--n_way_mtrain', type=int, help="""used if the number of ways is 
        different between meta-train and meta-test""", default=60)
    argparser.add_argument('--k_support', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_query', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--batch_size', type=int, help="""meta batch size, namely task num, 
        each epoch contains 6 batches""", default=32)
    argparser.add_argument('--image_size', type=int, help='size of input images of neural networks', default=32)
    argparser.add_argument('--epochs', type=int, help='how many epochs in total', default=300)
    argparser.add_argument('--seed', type=int, help='random seed', default=68)
    argparser.add_argument('--lr', type=float, default=0.0005, help="""learning rate""")
    argparser.add_argument('--use_lr_scheduler', action="store_true", default=False, help="""
        whether use LR scheduler as the original paper""")
    argparser.add_argument('--nb_test_episodes', type=int, help="""how many episodes to use to 
        compute the meta-test accuracy""", default=1000)
    argparser.add_argument('--nb_batches_to_preload', type=int, default=10)
    args = argparser.parse_args()
    
    t0_overall = time.time()

    project_name = "OmniPrint-{}_Proto".format(os.path.basename(args.dataset))
    group_name = "{}-way-{}-shot".format(args.n_way, args.k_support)
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
                                        n_way_mtrain=args.n_way_mtrain,
                                        k_support=args.k_support,
                                        k_query=args.k_query,
                                        image_size=args.image_size,
                                        device=device,
                                        msplit="32,8,14",
                                        nb_batches_to_preload=args.nb_batches_to_preload)

    net = Backbone().to(device)

    wandb.run.summary["trainable_parameters_count"] = count_trainable_parameters(net)
    
    env_info = get_torch_gpu_environment()
    for k, v in env_info.items():
        wandb.run.summary[k] = v
    
    optim = torch.optim.Adam(params=net.parameters(), lr=args.lr)

    if args.use_lr_scheduler:
        # 11 * 6 * 32 = 2112 > 2000
        # half the learning rate every 2112 episodes
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=0.5, step_size=11)

    checkpoints_dir = "model_checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    timestamp = str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))
    best_model = [-1, os.path.join(checkpoints_dir, 
        "best_model_{}_{}_{}.pth".format(project_name, group_name, timestamp))] # (score, path)

    log = []
    for epoch in range(args.epochs):
        mtrain(dataloader, net, device, optim, args.k_query, epoch, log)
        if args.use_lr_scheduler:
            lr_scheduler.step()
        if epoch % 5 == 4:
            mval(dataloader, net, device, args.k_query, epoch, log, best_model)

    mtest(dataloader, device, args.k_query, log, best_model, nb_test_episodes=args.nb_test_episodes)

    df = pd.DataFrame(log) 
    t_overall = time.time() - t0_overall 

    #timestamp = str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))
    #df.to_csv("maml_omniglot_like_log_{}.csv".format(timestamp))

    wandb.run.summary["overall_computation_time"] = t_overall


def mtrain(dataloader, net, device, optim, k_query, epoch, log):
    """
    one epoch of meta-train
    """
    net.train()
    #nb_batches_per_epoch_mtrain = dataloader.datasets["train"].shape[0] // dataloader.batch_size

    # if n_way_mtrain=5, each epoch iterates through "every" class with replacement
    ## e.g. 6 * 32 = 192 episodes
    nb_batches_per_epoch_mtrain = 6  

    for batch_idx in range(nb_batches_per_epoch_mtrain):
        start_time = time.time()
        # Sample one batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = dataloader.next("train")

        batch_size, _, _, _, _ = x_spt.size()

        qry_losses = []
        qry_accs = []
        
        for i in range(batch_size):
            # one meta-train episode
            optim.zero_grad()

            spt_sz = x_spt[i].shape[0]
            x_feature = net(torch.cat((x_spt[i], x_qry[i]), dim=0))
            x_feature_spt, x_feature_qry = x_feature[:spt_sz], x_feature[spt_sz:]
            loss, acc = prototypical_loss(x_feature_spt, y_spt[i], x_feature_qry, y_qry[i], k_query, device)
            
            loss.backward()
            optim.step()
            qry_losses.append(loss.item())
            qry_accs.append(acc.item())

        qry_losses, qry_losses_conf = compute_mean_and_confidence_interval(qry_losses)
        qry_accs, qry_accs_conf = compute_mean_and_confidence_interval(qry_accs)
        qry_accs *= 100
        qry_accs_conf *= 100

        i = epoch + float(batch_idx + 1) / nb_batches_per_epoch_mtrain
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}')

        log.append({
            'epoch': i,
            'mtrain_query_loss': qry_losses,
            'mtrain_query_accuracy': qry_accs,
            'mtrain_batch_computation_time': iter_time,
            'mtrain_query_loss_confidence_interval': qry_losses_conf,
            'mtrain_query_accuracy_confidence_interval': qry_accs_conf
        })
        wandb.log({"mtrain_query_loss": qry_losses, "mtrain_query_accuracy": qry_accs, 
            "epoch": i, "mtrain_batch_computation_time": iter_time, 
            "mtrain_query_loss_confidence_interval": qry_losses_conf,
            "mtrain_query_accuracy_confidence_interval": qry_accs_conf})


def mval(dataloader, net, device, k_query, epoch, log, best_model):
    """
    one epoch of meta-validation
    """
    net.eval()
    #nb_batches_per_epoch_mval = dataloader.datasets["val"].shape[0] // dataloader.batch_size

    # if n_way_mtrain=5, each epoch iterates through "every" class with replacement
    ## e.g. 6 * 32 = 192 episodes
    nb_batches_per_epoch_mval = 6  
    
    qry_losses = []
    qry_accs = []

    start_time = time.time()

    with torch.no_grad():
        for batch_idx in range(nb_batches_per_epoch_mval):
            x_spt, y_spt, x_qry, y_qry = dataloader.next("val")

            batch_size, _, _, _, _ = x_spt.size()

            qry_losses = []
            qry_accs = []
            
            for i in range(batch_size):
                # one meta-test episode
                spt_sz = x_spt[i].shape[0]
                x_feature = net(torch.cat((x_spt[i], x_qry[i]), dim=0))
                x_feature_spt, x_feature_qry = x_feature[:spt_sz], x_feature[spt_sz:]
                loss, acc = prototypical_loss(x_feature_spt, y_spt[i], x_feature_qry, y_qry[i], k_query, device)
                
                qry_losses.append(loss.item())
                qry_accs.append(acc.item())

            qry_losses, qry_losses_conf = compute_mean_and_confidence_interval(qry_losses)
            qry_accs, qry_accs_conf = compute_mean_and_confidence_interval(qry_accs)
            qry_accs *= 100
            qry_accs_conf *= 100
    
    print(f'[Epoch {epoch+1:.2f}] Validation Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}')
    
    epoch_time = time.time() - start_time

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
        torch.save(net.state_dict(), best_model[1])
        wandb.run.summary["best_mval_query_accuracy"] = qry_accs
        wandb.run.summary["best_mval_query_accuracy_confidence_interval"] = qry_accs_conf


def mtest(dataloader, device, k_query, log, best_model, nb_test_episodes=1000):
    """
    meta-test of the best model on meta-validation episodes
    """
    net = Backbone()
    net.load_state_dict(torch.load(best_model[1]))
    net.to(device)

    net.eval()
    
    qry_accs = []

    start_time = time.time()

    with torch.no_grad():
        while True:
            x_spt, y_spt, x_qry, y_qry = dataloader.next("test")

            batch_size, _, _, _, _ = x_spt.size()

            for i in range(batch_size):
                # one meta-test episode
                spt_sz = x_spt[i].shape[0]
                x_feature = net(torch.cat((x_spt[i], x_qry[i]), dim=0))
                x_feature_spt, x_feature_qry = x_feature[:spt_sz], x_feature[spt_sz:]
                loss, acc = prototypical_loss(x_feature_spt, y_spt[i], x_feature_qry, y_qry[i], k_query, device)
                
                qry_accs.append(acc.item())

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
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
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
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


def compute_prototypes(x_feature_spt, y_spt, classes):
    """
    computes the barycenter of each class using support examples
    """
    indices = list(map(lambda c: y_spt.eq(c).nonzero().squeeze(1), classes))
    return torch.stack([x_feature_spt[idx].mean(0) for idx in indices])


def group_query_features_by_class(x_feature_qry, y_qry, classes):
    indices = torch.stack(list(map(lambda c: y_qry.eq(c).nonzero(), classes))).view(-1)
    return x_feature_qry[indices]


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(x_feature_spt, y_spt, x_feature_qry, y_qry, k_query, device):
    classes = torch.unique(y_spt.to("cpu"))
    n_way = len(classes)

    prototypes = compute_prototypes(x_feature_spt, y_spt, classes)
    x_feature_qry_grouped_by_class = group_query_features_by_class(x_feature_qry, y_qry, classes)
    dists = euclidean_dist(x_feature_qry_grouped_by_class, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, k_query, -1)

    target_inds = torch.arange(0, n_way).to(device)
    target_inds = target_inds.view(n_way, 1, 1)
    target_inds = target_inds.expand(n_way, k_query, 1).long()
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    
    return loss_val,  acc_val


if __name__ == '__main__':
    main()





