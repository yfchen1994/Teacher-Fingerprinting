# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint_sequential
from model_stealing.query_blackbox import KnockoffDataset
import torchvision
import copy
from utils import *
from models.TransferLearningModel import TransferLearningModel
from models.PretrainedModel import PretrainedModel

def train(model, dataset, epochs=20, batch_size=128, learning_rate=0.01):
    """
    Train the student model.

    Args:
        model: the model to train.
        dataset: the training dataset. < 0.5)
        epochs: defaults to 20.
        batch_size: defaults to 128.

    Returns:
        The trained student model.
    """
    scaler = torch.cuda.amp.GradScaler()

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=12,
                                             pin_memory=True,
                                             shuffle=True)
    model.setup_trainable_params()
    model.train()
    model.cuda()

    if model.is_multilabel_task:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    print("Learning rate: {}".format(learning_rate))
    device = next(model.classifier.parameters()).device

    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            for param in model.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss = loss.item()
            if model.is_multilabel_task:
                preds = (outputs > 0.5).float()
                corrects_per_label = torch.abs(preds - labels.data) < 0.5
                running_corrects += torch.sum(
                    torch.sum(corrects_per_label, 1) / 40.)
            else:
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
            print('Epoch:%d, Batch:%d (Batch_size %d), Loss:%.3f' %
                  (epoch + 1, i + 1, batch_size, running_loss),
                  end='\r')

        accuracy = running_corrects * 1. / len(dataloader.dataset)
        print('Epoch:%d, Batch:%d, Loss: %.3f, accuracy: %.2f%%' %
              (epoch + 1, i + 1, running_loss, 100 * accuracy))
    return model


def train_transfer_learning_model(model, dataset_config, training_config,
                                  model_path):
    """
    Train the transfer learning model.

    Args:
        model: the transfer learning model to train.
        dataset_config: the dataset configuration.
        training_config: the training configuration.
        model_path: where to save the trained transfer learning model.

    Returns:
        The trained transfer learning model.
    """
    # Set up the dataset
    dataset = get_dataset_by_config(dataset_config)
    # Read the training parameters.
    epochs = training_config['epochs']
    batch_size = training_config['batch_size']
    if 'learning_rate' in training_config.keys():
        learning_rate = training_config['learning_rate']
    else:
        learning_rate = 0.01
    # Train the transfer learning model
    model = train(model,
                  dataset,
                  epochs=epochs,
                  batch_size=batch_size,
                  learning_rate=learning_rate)
    return model


def evaluate_model(model, dataset):
    """Evaluate the transfer learning model.

    Args:
        model: the transfer learning model. 
        dataset: the validation/testing dataset. 

    Returns:
        (testing accuracy,
         the number of correct prediction,
         the number of predictions.)
    """
    model.cuda()
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             pin_memory=True,
                                             num_workers=12,
                                             shuffle=False)
    running_corrects = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        if model.is_multilabel_task:
            preds = (outputs > 0.5).float()
            corrects_per_label = torch.abs(preds - labels.data) < 0.5
            running_corrects += torch.sum(
                torch.sum(corrects_per_label, 1) / 40.)
        else:
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        test_accuracy = running_corrects * 1. / len(dataloader.dataset)
    return test_accuracy, running_corrects, len(dataloader.dataset)


def get_transfer_learning_model(config_file, model_dir=''):
    """
    Obtain the transfer learning model according to the configuration file.
    (YAML format)

    Args:
        config_file: the path of the model configuration file.
        model_dir: the directory to save the learning mdoel.

    Returns:
        The trained transfer learning model.
    """
    configs = load_config(config_file)

    if len(model_dir) == 0:
        # If model_dir is not set by the user, then read
        # model_dir from the configuration file.
        model_dir = configs['model_dir']

    # Build up the model.
    model_config = configs['model_config']
    model = build_model_by_config(model_config)

    def _get_model_path(model_dir, model_config):
        # The model path is organized as the following format:
        # ./model_dir/{pretrained model name}_{fcn sctrucutre}_{tune conv?}.pkl
        pretrained_info = model_config['pretrained_model']
        fcn_info = ('_').join(
            [str(num) for num in model_config['fcn_neuron_num']])
        fine_tune_info = 'conv_tuned' if len(model.fine_tune_block_idx) > 0 \
                         else 'conv_fixed'
        model_file_name = '{}_{}_{}.pkl'.format(pretrained_info, fcn_info,
                                                fine_tune_info)
        model_path = os.path.join(model_dir, model_file_name)
        return model_path

    # Where is the transfer learning model.
    model_path = _get_model_path(model_dir, model_config)

    if os.path.exists(model_path):
        # If the model already exists, load the model.
        print("Model exists")
        print(model_path)
        model = load_model(model, model_path)
    else:
        print("Model does not exist")
        print(model_path)
        # If the model does not exist, train and save the model.
        dataset_config = configs['dataset_config']
        training_config = configs['training_config']
        seed = time.time()
        torch.manual_seed(seed)
        model = train_transfer_learning_model(model, dataset_config,
                                              training_config, model_path)
        save_model(model, model_path)
        del model
        torch.cuda.empty_cache()
        model = build_model_by_config(model_config)
        model = load_model(model, model_path)

        def _log_model(model_path, torch_seed):
            """Logging the training process.

            Args:
                model_path: the path to save the trained model.
                torch_seed: the random seed used for training the model.
                            It may be useful for reproducing the model.
            """
            log = {}
            log_file_path = model_path[:-4] + '.log'
            log['model_path'] = model_path
            log['model_cfg_path'] = config_file
            log['torch_seed'] = torch_seed

            if dataset_config['dataset_name'] == 'CIFAR100':
                other_config = {'train': False}
            elif dataset_config['dataset_name'] == 'CIFAR10':
                other_config = {'train': False}
            elif dataset_config['dataset_name'] == 'MNIST':
                other_config = {'train': False}
            else:
                other_config = {'split': 'test'}

            test_dataset = get_dataset_by_config(
                dataset_config, other_config_to_reset=other_config)
            test_accuracy, corrects, total = evaluate_model(
                model, test_dataset)

            log['evaluation_result'] = 'Test accuracy: {:.2f}% ({}/{})'\
                                       .format(100. * test_accuracy,
                                               corrects,
                                               total)
            if len(model.fine_tune_block_idx) > 0:
                log['fine_tune_block_idx'] = model.fine_tune_block_idx

            with open(log_file_path, 'w') as f:
                print(log)
                json.dump(log, f)

        if dataset_config['dataset_name'] == 'Knockoff':
            # For the knockoff model, it will be evaluated in a
            # different way.
            return model
        else:
            _log_model(model_path, torch_seed=seed)

    return model.cuda()


def build_model_by_config(model_config):
    """
    Build up the transfer learning model according to the model configuration.

    Args:
        model_config: the model configuration.

    Returns:
        The model built up by to the model configuration.
    """

    MULTILABEL_TASK_KEY = 'is_multilabel_task'
    FINE_TUNE_KEY = 'fine_tune_block_idx'
    PRETRAINED_KEY = 'pretrained'

    model_name = model_config['pretrained_model']
    fcn_neuron_num = model_config['fcn_neuron_num']
    input_size = model_config['input_size']

    if MULTILABEL_TASK_KEY in model_config.keys():
        is_multilabel_task = model_config[MULTILABEL_TASK_KEY]
    else:
        is_multilabel_task = False

    if 'fine_tune_block_idx' in model_config.keys():
        fine_tune_block_idx = model_config['fine_tune_block_idx']
    else:
        fine_tune_block_idx = False

    if 'selected_block_amount' in model_config.keys():
        selected_block_amount = model_config['selected_block_amount']
    else:
        selected_block_amount = 0

    if FINE_TUNE_KEY in model_config.keys():
        fine_tune_block_idx = model_config[FINE_TUNE_KEY]
        print("{}:{}".format(FINE_TUNE_KEY, str(fine_tune_block_idx)))
    else:
        fine_tune_block_idx = []

    if PRETRAINED_KEY in model_config.keys():
        pretrained = model_config[PRETRAINED_KEY]
    else:
        pretrained = True

    # Remove the classifiers from the pretrained model.
    pretrained_model = PretrainedModel(
        pretrained_model_name=model_name,
        pretrained=pretrained,
        selected_block_amount=selected_block_amount)
    # Build up the transfer learning model.
    model = TransferLearningModel(model_pretrain=pretrained_model,
                                  fcn_neuron_num=fcn_neuron_num,
                                  input_size=input_size,
                                  is_multilabel_task=is_multilabel_task,
                                  fine_tune_block_idx=fine_tune_block_idx)
    return model


def get_dataset_by_config(dataset_config, other_config_to_reset=None):
    """
    Prepare the dataset according to the dataset configuration.

    Args:
        dataset_config: the dataset configuration.
        other_config_to_reset: other config set manually set.

    Returns:
        A PyTorch dataset.
    """
    dataset_name = dataset_config['dataset_name']

    # Setup the transform
    resized_size = dataset_config['resized_size']
    transforms_list = [
        torchvision.transforms.Resize(resized_size),
        torchvision.transforms.ToTensor()
    ]

    if dataset_name == 'MNIST':
        # For the MNIST dataset, convert the grayscale image to RGB.
        transforms_list.append(torchvision.transforms.\
                              Lambda(lambda x: x.repeat(3, 1, 1)))

    if 'normalize' in dataset_config.keys():
        mean = dataset_config['normalize']['mean']
        std = dataset_config['normalize']['std']
        transforms_list.append(transforms.Normalize(mean=mean, std=std))

    transform = torchvision.transforms.Compose(transforms_list)

    if other_config_to_reset:
        other_config = other_config_to_reset
    else:
        other_config = dataset_config['other_config']

    # Compare the dataset
    if dataset_config['dataset_name'] == 'KnockoffDataset':
        knockoff_dataset_dir = dataset_config['dataset_dir']
        return KnockoffDataset(knockoff_dataset_dir, transform=transform)

    dataset = load_dataset(dataset_name=dataset_name,
                           kwargs=other_config,
                           transform=transform)

    return dataset


def create_target_models(config_dir):
    """Try to generate a series of target models according to model configurations
    under the directory config_dir.

    Args:
        config_dir: the directory saving the model configuration files.
    """
    for config_file in os.listdir(config_dir):
        if config_file.split('.')[-1] != 'yaml':
            continue
        config_file_path = os.path.join(config_dir, config_file)
        print('*' * 10)
        print(config_file_path)
        try:
            model = get_transfer_learning_model(config_file_path)
        except:
            print('Error: {}'.format(sys.exc_info()))
            # *Warining:*
            # If some errors occur, will continue to generate the remaining models.
            # It means that sometimes you may not obtain all the models wanted.
            continue
