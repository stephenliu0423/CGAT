import argparse
import numpy as np
from dataloader import load_data
from train import train
import random
import torch as t


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='music',
                    help='which dataset to use')
parser.add_argument('--dim', type=int, default=64,
                    help='dimension of entity and relation embeddings') 
parser.add_argument('--l2_weight_rs', type=float, default=0.00005,
                    help='weight of the l2 regularization term')
parser.add_argument('--lr_rs', type=float,
                    default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--n_epochs', type=int, default=50,
                    help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=16,
                    help='fixed size of user historical items')
parser.add_argument('--use_cuda', type=bool, default=True,
                    help='use cuda.')
parser.add_argument('--n_neighbor', type=int, default=4,
                    help='fixed size of neighbor entities')
parser.add_argument('--kg_weight', type=float,
                    default=0.0001, help='weight of regularization')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate')

np.random.seed(2019)
random.seed(2019)
t.manual_seed(2019)
t.cuda.manual_seed_all(2019)
args = parser.parse_args()

show_loss = True
show_topk = True
data_info = load_data(args)
train(args, data_info, show_loss, show_topk)
