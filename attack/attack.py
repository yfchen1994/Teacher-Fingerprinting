# -*- coding: utf-8 -*-

import os
import sys
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint_sequential
import torchvision
from utils import *
from attack.generate_queries import obtain_queries

def attack(target_model,
           query_config,
           dataset_config,
           attack_config,
           if_output_img=True,
           synthetic_inputs_cached={},
           probing_inputs_cached={}):
    """The teacher model fingerprinting attack.

    Args:
        target_model: the victim student model.
        query_config: the configuration of queries. 
        dataset_config: the configuration of attack datasets 
                        (where the probing inputs come from).
        attack_config: the configuration of the query generation algorithm.
        if_output_img: whether to save the queries as images.
        synthetic_inputs_cached: already generated synthetic inputs.
        probing_inputs_cached: already used probing inputs.

    Returns:
        _type_: _description_
    """
    torch.cuda.empty_cache()

    attack_model_name = query_config['attack_model_name']
    dataset_name = dataset_config['dataset_name']
    dict_key = "{}_{}".format(attack_model_name, dataset_name)

    if dict_key in synthetic_inputs_cached.keys():
        synthetic_inputs = synthetic_inputs_cached[dict_key]
        probing_inputs = probing_inputs_cached[dict_key]
    else:
        target_model.eval()
        for params in target_model.parameters():
            params.requires_grad = False
        target_model = target_model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        # Obtain the attack queries.
        synthetic_inputs, probing_inputs = obtain_queries(query_config,
                                                           dataset_config,
                                                           attack_config,
                                                           if_output_img)
        synthetic_inputs = torch.FloatTensor(synthetic_inputs).cpu()
        probing_inputs = torch.FloatTensor(probing_inputs).cpu()

        synthetic_inputs_cached[dict_key] = synthetic_inputs
        probing_inputs_cached[dict_key] = probing_inputs
        target_model = target_model.cuda()

    # Disable the dropout layer.
    target_model.eval()

    # Send the queries
    y_synthetic = target_model(synthetic_inputs.cuda())
    y_probing = target_model(probing_inputs.cuda())

    # Compute the heuristics.
    fingerprinting_vector, match_cnt, support_cnt = get_fingerprinting_vector(y_synthetic, y_probing)
    entropy = get_entropy(y_synthetic, y_probing)
    # The following two metrics are not used in our original paper.
    l1_distance = get_l1_distance(y_synthetic, y_probing)
    cosine_similarity = get_cosine_similarity(y_synthetic, y_probing)

    return (fingerprinting_vector,
            match_cnt,
            support_cnt,
            l1_distance,
            cosine_similarity,
            entropy,
            synthetic_inputs_cached,
            probing_inputs_cached)

def get_fingerprinting_vector(y1, y2):
    """Calculate the fingerprinting vector for an attack.

    Args:
        y1: the predicted labels of the synthetic inputs.
        y2: the predicted labels of the probing inputs.

    Returns:
        (The fingerprinting vector,
         the size of the matching set,
         the size of the supporting set.)
    """
    assert len(y1) == len(y2)
    _, y1 = torch.max(y1, 1)
    _, y2 = torch.max(y2, 1)
    match_cnt = torch.sum(y1==y2)
    similarity = match_cnt * 1. / len(y1)

    # Get the size of supporting set
    y_matched = y1[y1==y2]
    if len(y_matched) == 0:
        return (0, 0, 0)
    removed_cnt, removed_element = get_the_most_frequent_element(y_matched)
    support_cnt = len(y_matched) - removed_cnt

    return (similarity.item(), match_cnt.item(), support_cnt.item())

def get_the_most_frequent_element(y):
    """Find the most frequent element in the matching set.
    This function is used for building up the supporting set.

    Args:
        y: the matching set.

    Returns:
        The most frequent element in the matching set.
    """
    max_cnt = 0
    most_frequent_element = 0
    for i in y.unique():
        cnt = torch.sum(y==i)
        if max_cnt < cnt:
            max_cnt = cnt
            most_frequent_element = i
    return (max_cnt, most_frequent_element)

def get_l1_distance(y1, y2):
    """Calculate the l1-distance heuristic for an attack.
    *We don't use this heuristic in our original paper. 
    You may find it useful for your own work :)*

    Args:
        y1: the predicted labels of the synthetic inputs.
        y2: the predicted labels of the probing inputs.

    Returns:
        The l1-distance heuristic for an attack. 
    """
    assert len(y1) == len(y2)
    distance = torch.nn.functional.l1_loss(y1, y2)
    return distance.item()

def get_cosine_similarity(y1, y2):
    """Calculate the cosine similarity heuristic for an attack.
    *We don't use this heuristic in our original paper. 
    You may find it useful for your own work :)*

    Args:
        y1: the predicted labels of the synthetic inputs.
        y2: the predicted labels of the probing inputs.

    Returns:
        The consine similarity heuristic for an attack. 
    """
    assert len(y1) == len(y2)
    cosine_similarity = torch.mean(torch.nn.functional.cosine_similarity(y1,y2))
    return cosine_similarity.item()

def get_entropy(y1, y2):
    """Calculate the entropy heuristic for an attack.

    Args:
        y1: the predicted labels of the synthetic inputs.
        y2: the predicted labels of the probing inputs.

    Returns:
        The entropy heuristic for an attack. 
    """
    assert len(y1) == len(y2)
    _, y1 = torch.max(y1, 1)
    _, y2 = torch.max(y2, 1)
    matched = y1[y1 == y2]
    if len(matched) == 0:
        # If there is no matched pairs, set the entropy as 0.
        entropy = 0
    else:
        entropy = calculate_entropy(matched)
    return entropy

def calculate_entropy(y):
    """Calculate the entropy heuristic.

    Args:
        y: probabilities.

    Returns:
        The entropy heuristic.
    """
    p = calculate_probs(y)
    entropy = torch.distributions.Categorical(probs=p).entropy()
    return entropy.item()

def calculate_probs(y):
    """Calculating the probabilities (i.e., frequences) of each class.

    Args:
        y: predicted labels.

    Returns:
        The probabilities of each class.
    """
    _, counts = torch.unique(y, return_counts = True)
    p = counts / len(y)
    return p
