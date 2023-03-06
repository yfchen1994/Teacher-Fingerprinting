# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from models.transfer_learning import create_target_models
import torch


def train_transfer_learning_model(root_config_dir):
    """Generate transfer learning modesl by configuration files
       under ``root_config_dir.''

    Args:
        root_dir: the directory saving 
        configuration files of transfer learning models.
    """
    for root, dirs, files in os.walk(root_config_dir):
        if len(dirs) == 0:
            sub_config_dir = root
            print(sub_config_dir)
            create_target_models(sub_config_dir)
        else:
            for sub_dir in dirs:
                sub_config_dir = os.path.join(root, sub_dir)
                print(sub_config_dir)
                create_target_models(sub_config_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description="Training transfer learning mdoels.")
    parser.add_argument("--config_root_dir",
                        type=str,
                        help="Where the model configuration files are.")
    args = parser.parse_args()
    train_transfer_learning_model(args.config_root_dir)
