# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adapted by Haozhe Sun in order to test OmniPrint datasets
The modifications include:
* Modify logging systems, e.g. wandb
* Change the specification of training, e.g. learning rate,
  epoch definition, epoch count, the way to interact with 
  validation set and test set (the best validation model during 
  training is tested on test set), etc.
* The algorithm itself is untouched but the code is refactored
* Some other minor but necessary modifications for the formatting or utilities

This code is modified from: 
https://github.com/facebookresearch/higher/blob/master/examples/maml-omniglot.py
which is modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py

The original MAML paper:
https://arxiv.org/abs/1703.03400
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

import higher

import os
import sys
import datetime
sys.path.append(os.path.join(os.pardir, "omniprint"))
sys.path.append(os.path.join(os.pardir, "omniprint", "dataloader"))

from dataloader import OmniglotLikeZDataloader

import wandb

from utils import *


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='which dataset to test', 
                            default='../omniprint/omniglot_like_datasets/meta3')
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_support', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_query', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--batch_size', type=int, help="""meta batch size, namely task num, 
        each epoch contains 6 batches""", default=32)
    argparser.add_argument('--image_size', type=int, help='size of input images of neural networks', default=32)
    argparser.add_argument('--inner_steps', type=int, help='number of inner steps in each loop', default=5)
    argparser.add_argument('--meta_lr', type=float, help='learning rate of the meta optimizer', default=1e-3)
    argparser.add_argument('--epochs', type=int, help='how many epochs in total', default=60)
    argparser.add_argument('--seed', type=int, help='random seed', default=68)
    argparser.add_argument('--nb_test_episodes', type=int, help="""how many episodes to use to 
        compute the meta-test accuracy""", default=1000)
    argparser.add_argument('--n_jobs_knn', type=int, help='number of processes for K-nearest-neighbor search', default=None)
    args = argparser.parse_args()
    
    t0_overall = time.time()

    project_name = "OmniPrint-Z-{}_MAML".format(os.path.basename(args.dataset))
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

    dataloader = OmniglotLikeZDataloader(data_path=args.dataset, 
                                         batch_size=args.batch_size,
                                         n_way=args.n_way,
                                         k_support=args.k_support,
                                         k_query=args.k_query,
                                         image_size=args.image_size,
                                         device=device,
                                         msplit="32,8,14",
                                         n_jobs_knn=args.n_jobs_knn,
                                         z_metadata=["shear_x", "rotation"])

    # in_channels=3 because of RGB images
    net = Backbone(args.n_way).to(device)

    wandb.run.summary["trainable_parameters_count"] = count_trainable_parameters(net)
    
    env_info = get_torch_gpu_environment()
    for k, v in env_info.items():
        wandb.run.summary[k] = v

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam(net.parameters(), lr=args.meta_lr)

    checkpoints_dir = "model_checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    timestamp = str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))
    best_model = [-1, os.path.join(checkpoints_dir, 
        "best_model_{}_{}_{}.pth".format(project_name, group_name, timestamp))] # (score, path)

    log = []
    for epoch in range(args.epochs):
        mtrain(dataloader, net, device, meta_opt, epoch, args.inner_steps, log)
        if epoch % 5 == 4:
            mval(dataloader, net, device, epoch, args.inner_steps, log, best_model)

    mtest(dataloader, device, args.inner_steps, log, best_model, args.n_way, nb_test_episodes=args.nb_test_episodes)

    df = pd.DataFrame(log) 
    t_overall = time.time() - t0_overall 

    #timestamp = str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))
    #df.to_csv("maml_omniglot_like_log_{}.csv".format(timestamp))

    wandb.run.summary["overall_computation_time"] = t_overall
    


def mtrain(dataloader, net, device, meta_opt, epoch, inner_steps, log):
    net.train()
    #nb_batches_per_epoch_mtrain = dataloader.datasets["train"].shape[0] // dataloader.batch_size

    # if n_way=5, each epoch iterates through "every" class with replacement
    ## e.g. 6 * 32 = 192 episodes
    nb_batches_per_epoch_mtrain = 6  
    
    for batch_idx in range(nb_batches_per_epoch_mtrain):
        start_time = time.time()
        # Sample one batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = dataloader.next("train")

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                for _ in range(inner_steps):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = fnet(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach().cpu().item())
                qry_acc = (qry_logits.argmax(dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward()

        meta_opt.step()
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


def mval(dataloader, net, device, epoch, inner_steps, log, best_model):
    net.train()
    #nb_batches_per_epoch_mval = dataloader.datasets["val"].shape[0] // dataloader.batch_size

    # if n_way=5, each epoch iterates through "every" class with replacement
    ## e.g. 6 * 32 = 192 episodes
    nb_batches_per_epoch_mval = 6
    
    qry_losses = []
    qry_accs = []

    start_time = time.time()

    for batch_idx in range(nb_batches_per_epoch_mval):
        x_spt, y_spt, x_qry, y_qry = dataloader.next("val")

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                for _ in range(inner_steps):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The query loss and acc induced by these parameters.
                qry_logits = fnet(x_qry[i]).detach()
                qry_loss = F.cross_entropy(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach().cpu().item())
                qry_accs.append((qry_logits.argmax(dim=1) == y_qry[i]).detach().float().mean().cpu().item())

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


def mtest(dataloader, device, inner_steps, log, best_model, n_way, nb_test_episodes=1000):
    net = Backbone(n_way)
    net.load_state_dict(torch.load(best_model[1]))
    net.to(device)

    net.train()
    
    qry_accs = []

    start_time = time.time()

    while True:
        x_spt, y_spt, x_qry, y_qry = dataloader.next("test")

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                for _ in range(inner_steps):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                qry_logits = fnet(x_qry[i]).detach()
                qry_accs.append((qry_logits.argmax(dim=1) == y_qry[i]).detach().float().mean().cpu().item())

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
    def __init__(self, n_way):
        super().__init__()
        self.sequential = nn.Sequential(
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
            Flatten(),
            nn.Linear(64, n_way)
        )

    def forward(self, x):
        return self.sequential(x)



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


if __name__ == '__main__':
    main()


