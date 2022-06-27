import argparse
import os
from os.path import join
import sys
import torch
import utils
import multiprocessing

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--test_u_batch_size', type=int, default=100,
                        help="the batch size of users for testing")
parser.add_argument('--multicore', type=int, default=1,
                        help='whether we use multiprocessing or not in test')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
parser.add_argument('--history_len', type=int, default=10,
                        help='length of memorization history')
parser.add_argument('--meta_lr', type=float, default=0.0005,
                        help="the learning rate of meta-learning procedure")
parser.add_argument('--schedule_lr', type=float, default=0.005,
                        help="the learning rate of scheduler")
parser.add_argument('--model', type=str, default='lgn', help='backbone model')
parser.add_argument('--eval_freq', type=int, default=10, help='validation frequency')
parser.add_argument('--stop_step', type=int, default=4, help='for early stop')
parser.add_argument('--cuda_device', type=str, default='0')
parser.add_argument('--seed', type=int, default=2020, help='random seed')
parser.add_argument('--topks', nargs='?', default="[5,20]",
                        help="@k test list")
parser.add_argument('--dataset', type=str, default='ml100k',
                    help="available datasets: [yelp, ml100k, adressa]")
parser.add_argument('--epochs', type=int, default=1000)


# ============= Params for LightGCN =============== #
parser.add_argument('--latent_dim_rec', type=int, default=64,
                    help='the embedding size of lightGCN')
parser.add_argument('--lightGCN_n_layers', type=int, default=3,
                    help='the layer num of lightGCN')
parser.add_argument('--dropout', type=int, default=0,
                    help="using the dropout or not")
parser.add_argument('--A_split', type=bool, default=False)
parser.add_argument('--keep_prob', type=float, default=0.6,
                    help="the batch size for bpr loss training procedure")
parser.add_argument('--A_n_fold', type=int, default=100,
                    help="the fold num used to split large adj matrix, like gowalla")

# ============= Params for LTW =============== #
parser.add_argument('--input', type=int, default=1, help='input size of LTW')
parser.add_argument('--hidden1', type=int, default=100, help='hidden size of LTW')
parser.add_argument('--output', type=int, default=1, help='output size of LTW')

# ============= Params for Scheduler =============== #
parser.add_argument('--schedule_type', type=str, default='gumbel', help='training strategy of scheduler: reinforce, gumbel')
parser.add_argument('--tau', type=float, default=1.0, help='temperature of gumbel softmax')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
ROOT_PATH = "./"
CODE_PATH = join(ROOT_PATH, 'code')
FILE_PATH = join(CODE_PATH, 'checkpoints')
sys.path.append(join(CODE_PATH, 'sources'))
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
topks = eval(args.topks)

log_file = f'./log/{args.dataset}_{args.model}_lr{args.lr}_metalr{args.meta_lr}_{args.schedule_type}_tau{args.tau}_schedule_{args.schedule_lr}.txt'

#log_file = f'./log/debug.txt'

f = open(log_file, 'w')
f.close()

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)