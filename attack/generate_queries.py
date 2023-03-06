# -*- coding: utf-8 -*-

import os
import gc
import PIL
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint_sequential
import torchvision
from models.PretrainedModel import PretrainedModel
from utils import check_directory, load_dataset, CustomImageFolder


def craft_queries(model,
                  input_size,
                  dataloader,
                  query_budget=100,
                  mean=None,
                  std=None,
                  learning_rate=0.001,
                  batch_size=16,
                  iters=1000,
                  eps=1e-8,
                  block_num=0):
    """Crafting synthetic queries.

    Args:
        model: the teacher model candidate. 
        input_size: the model's input size.
        dataloader: dataloader of the probing inputs.
        query_budget: the query budget. Defaults to 100.
        mean: the "mean" parameter used by the image preprocessor. Defaults to None.
        std: the "std" parameter used by the image preprocessor. Defaults to None.
        learning_rate: learning rate of the optimizer. Defaults to 0.001.
        batch_size: batch size of the image generation algorithms. Defaults to 16.
        iters: the number of optimization iterations for each batch. Defaults to 1000.
        eps (_type_, optional): _description_. Defaults to 1e-8.
        block_num: from which block to craft synthetic inputs. Defaults to 0.
    """

    MAX_VALUE = 1.
    MIN_VALUE = 0.
    old_loss = 100
    CHANNEL_N = input_size[0]

    if mean:
        min_ = (MIN_VALUE - np.array(mean)) / np.array(std)
        max_ = (MAX_VALUE - np.array(mean)) / np.array(std)
    else:
        min_ = MIN_VALUE * np.ones(CHANNEL_N)
        max_ = MAX_VALUE * np.ones(CHANNEL_N)

    def min_max_mapping(tmp_var, min, max):
        # Scaling the range of tmp_var from [0,1] to [min,max].
        for i in range(len(min)):
            tmp_var[:,
                    i, :, :] = (max[i] - min[i]) * tmp_var[:, i, :, :] + min[i]
        return tmp_var

    def reverse_min_max_mapping(tmp_var, min, max):
        # Revese operation of the min_max_mapping() function,
        # i.e., scaling the range of tmp_var from [min,max] to [0,1]
        for i in range(len(min)):
            tmp_var[:, i, :, :] = (tmp_var[:, i, :, :] - min[i]) / (max[i] -
                                                                    min[i])
        return tmp_var

    def _craft_queries_per_batch(probing_input, probing_feature, model,
                                 block_num, iters):
        """Generate synthetic queries per batch.

        Args:
            probing_input: probing inputs. 
            probing_feature: the intermediate features of the probing inputs.
            model: the teacher model candidate. 
            block_num: from which block to craft synthetic inputs. Defaults to 0.
            iters: the number of optimization iterations for each batch. Defaults to 1000.

        Returns:
            The synthetic inputs generated for this batch.
        """
        old_loss = 1e5

        # In our work, we initialize variable w as 0.
        # Another choice is to initilize w with random values.
        w = torch.zeros_like(probing_input).cuda()
        w.detach_()
        w.requires_grad = True
        optimizer = torch.optim.Adam([w], learning_rate)

        probing_feature.detach_()

        model.eval()

        for params in model.parameters():
            params.require_grad = False

        for i in range(iters):
            tmp_var = 0.5 * (torch.nn.Tanh()(w) + 1)
            tmp_var = min_max_mapping(tmp_var, min_, max_)
            output = model.pretrained_forward(tmp_var, block_num=block_num)
            loss = torch.nn.functional.mse_loss(output, probing_feature)
            current_loss = loss.item()
            print("Iters:{}, loss:{:.8f}".format(i + 1, current_loss),
                  end='\r')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print()
        synthetic_input = 0.5 * (torch.nn.Tanh()(w) + 1).detach()
        synthetic_input = min_max_mapping(synthetic_input, min_, max_)
        return synthetic_input

    reserved_num = 0

    synthetic_input = np.zeros((query_budget, ) + tuple(input_size))
    probing_input = np.zeros((query_budget, ) + tuple(input_size))

    for i, data in enumerate(dataloader, 0):
        probing_input_batch, _ = data
        with torch.no_grad():
            probing_feature_batch = model.pretrained_forward(
                probing_input_batch.cuda(), block_num=block_num)
        probing_feature_batch = probing_feature_batch.clone().float().cuda()
        synthetic_input_batch = _craft_queries_per_batch(
            probing_input_batch.float().cuda(), probing_feature_batch, model,
            block_num, iters)
        del probing_feature_batch
        torch.cuda.empty_cache()

        with torch.no_grad():
            synthetic_y = model.full_pretrained_model_forward(
                synthetic_input_batch)
            probing_y = model.full_pretrained_model_forward(
                probing_input_batch)
        # Get the top-1 label.
        _, synthetic_label = torch.max(synthetic_y, 1)
        _, probing_label = torch.max(probing_y, 1)
        # Convert to numpy
        synthetic_input_batch = synthetic_input_batch.cpu().numpy()
        probing_label = probing_label.cpu().numpy().reshape(-1)
        synthetic_label = synthetic_label.cpu().numpy().reshape(-1)
        # Select queries that activate matched predictions.
        cond = (probing_label == synthetic_label)
        synthetic_input_batch = synthetic_input_batch[np.where(cond)]
        new_reserved_num = np.min(
            [reserved_num + synthetic_input_batch.shape[0], query_budget])
        if new_reserved_num == reserved_num:
            continue

        synthetic_input[reserved_num:new_reserved_num] = synthetic_input_batch[
            0:new_reserved_num - reserved_num]
        probing_input[reserved_num:new_reserved_num] = probing_input_batch.cpu(
        ).numpy()[0:new_reserved_num - reserved_num]
        reserved_num = new_reserved_num
        print()
        print("Current # of selected inputs: {}".format(reserved_num))
        if reserved_num >= query_budget:
            break

    return (synthetic_input, probing_input)


def obtain_queries(query_config,
                   dataset_config,
                   attack_config,
                   if_output_img=True):

    QUERY_CFGS = ['query_img_dir', 'attack_model_name', 'query_budget']
    DATASET_CFGS = ['dataset_name', 'kwargs']
    ATTACK_CFGS = ['block_num']

    for key in QUERY_CFGS:
        if key not in query_config.keys():
            raise ValueError("Key {} is not included for \
                              the query configurations".format(key))

    query_img_root_dir = query_config['query_img_dir']
    attack_model_name = query_config['attack_model_name']
    query_budget = query_config['query_budget']

    for key in DATASET_CFGS:
        if key not in dataset_config.keys():
            raise ValueError("Key {} is not included for \
                              the dataset configurations".format(key))

    if ATTACK_CFGS[0] not in attack_config.keys():
        block_num = 0
    else:
        block_num = attack_config['block_num']

    dataset_name = dataset_config['dataset_name']

    target_img_sub_dir = 'target/{}_{}_{}_[block:{}]'\
                          .format(attack_model_name,
                                  dataset_name,
                                  query_budget,
                                  block_num)

    query_img_sub_dir = 'query/{}_{}_{}_[block:{}]'\
                          .format(attack_model_name,
                                  dataset_name,
                                  query_budget,
                                  block_num)
    query_img_dir = os.path.join(query_img_root_dir, query_img_sub_dir)
    target_img_dir = os.path.join(query_img_root_dir, target_img_sub_dir)

    # Build up the transform w.r.t. model configuration
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(attack_config['mean'],
                                         attack_config['std']),
    ])

    if if_output_img:
        ### Load queries from images
        if os.path.exists(query_img_dir):

            if 'effective_query_budget' in query_config:
                # We simulate an inference attack with n queries, where
                # n (e.g., the effective query length) <= query length.
                effective_query_budget = query_config['effective_query_budget']
            else:
                effective_query_budget = query_budget

            queries = load_query_from_imgs(query_img_dir, transform,
                                           effective_query_budget)
            target_x = load_query_from_imgs(target_img_dir, transform,
                                            effective_query_budget)
            return (queries, target_x)

    # Load the attack model
    attack_model = PretrainedModel(attack_model_name).cuda()
    attack_model.eval()
    for param in attack_model.parameters():
        param.require_grad = False
    input_size = attack_model.input_size

    # Load the data
    dataset = load_dataset(**dataset_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=attack_config['batch_size'],
        num_workers=12,
        shuffle=False)

    queries, target_x = craft_queries(model=attack_model,
                                      input_size=input_size,
                                      dataloader=dataloader,
                                      query_budget=query_budget,
                                      **attack_config)
    del attack_model
    torch.cuda.empty_cache()

    if if_output_img:
        if 'mean' in attack_config.keys():
            mean = attack_config['mean']
            std = attack_config['std']
        else:
            mean = None
            std = None

        check_directory(query_img_dir)
        save_query_img_from_numpy(queries, query_img_dir, mean, std)

        check_directory(target_img_dir)
        save_query_img_from_numpy(target_x, target_img_dir, mean, std)

        queries = load_query_from_imgs(query_img_dir, transform, query_budget)
        target_x = load_query_from_imgs(target_img_dir, transform,
                                        query_budget)
        return (queries, target_x)

    return (queries, target_x)


def load_query_from_imgs(query_img_dir, transform, query_budget):
    """Load attack queries from image files.

    Args:
        query_img_dir: the directory saving the images.
        transform: image transformation in the preprocessing stage.
        query_budget: the query budget.

    Returns:
        The attack queries.
    """
    imagef = CustomImageFolder(img_dir=query_img_dir, transform=transform)
    query_loader = torch.utils.data.DataLoader(imagef,
                                               batch_size=query_budget,
                                               num_workers=12,
                                               shuffle=False)
    for idx, img in enumerate(query_loader):
        query = img
        return query


def img_tensor_to_numpy(img_tensor, mean=None, std=None):
    """Revert Tensors to numpy arrays that present images.

    Args:
        img_tensor (_type_): PyTorch Tensors that present images.
        mean: the "mean" parameter used by the image preprocessor. Defaults to None.
        std: the "std" parameter used by the image preprocessor. Defaults to None.

    Returns:
        Numpy arrays that present images.
    """
    n_channel = len(mean)

    if type(img_tensor) is torch.Tensor:
        img_tensor = img_tensor.cpu().numpy()

    image_data = img_tensor.copy()

    if mean:
        for i in range(n_channel):
            image_data[:,i, :, :] = image_data[:,i, :, :] * std[i] + mean[i]

    image_data *= 255.
    image_data = image_data.transpose(0, 2, 3, 1)
    image_data = image_data.astype(np.uint8)
    return image_data


def save_query_img_from_numpy(query_tensor,
                              query_img_dir,
                              mean=None,
                              std=None):
    """Save the queries as images.

    Args:
        query_tensor: attack queries in the PyTorch Tensor format.
        query_img_dir: the directory to save the images.
        mean: the "mean" parameter used by the image preprocessor. Defaults to None.
        std: the "std" parameter used by the image preprocessor. Defaults to None.
    """
    image_data = img_tensor_to_numpy(query_tensor, mean, std)

    for i in range(image_data.shape[0]):
        img = PIL.Image.fromarray(image_data[i])
        img_path = os.path.join(query_img_dir, "{}.png".format(i))
        img.save(img_path)
