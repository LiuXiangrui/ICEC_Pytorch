# -- coding: utf-8 --**

import argparse
import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_epochs", type=int, default=1, help="eval model after `eval_epochs epochs")
    parser.add_argument("--save_directory", type=str, help="path of folder to save experimental data")

    parser.add_argument("--checkpoints", type=str, default=None, help="path of checkpoints")
    parser.add_argument("--pretrained", type=str, default=None, help="path of pretrained weights")

    parser.add_argument("--root", type=str, help="root path of dataset")
    parser.add_argument("--train_dataset", type=str, default='train', help='the folder name of the training sub folder')
    parser.add_argument("--eval_dataset", type=str, default='test', help='the folder name of the testing sub folder')

    parser.add_argument("--gpu", action='store_true', default=True, help="use gpu or cpu")
    parser.add_argument("--batch", type=int, default=2, help="batch size")
    parser.add_argument("--max_epoch", type=int, default=1000, help="max training epochs")
    parser.add_argument("--training_slices", type=int, default=15, help="slices for training")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_decay_milestone", type=int, nargs='+', help="lr decays milestone")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1, help="factor of learning rate decay")

    args = parser.parse_args()
    return args


def init() -> tuple:
    args = parse_args()

    # create directory for recording
    experiment_dir = Path(args.save_directory)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = Path(str(experiment_dir) + '/' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    experiment_dir.mkdir(exist_ok=True)

    # create directory for checkpoints
    ckpt_dir = experiment_dir.joinpath("Checkpoints/")
    ckpt_dir.mkdir(exist_ok=True)

    # initialize tensorboard
    tb_dir = experiment_dir.joinpath('Tensorboard/')
    tb_dir.mkdir(exist_ok=True)
    tensorboard = SummaryWriter(log_dir=str(tb_dir), flush_secs=30)

    return args, ckpt_dir, tensorboard
