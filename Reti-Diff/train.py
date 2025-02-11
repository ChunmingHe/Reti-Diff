# flake8: noqa
import os.path as osp
from train_pipeline import train_pipeline

import archs
import data
import models
import losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
