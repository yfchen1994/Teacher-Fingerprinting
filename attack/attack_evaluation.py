# -*- coding: utf-8 -*-

import os
import sys
import json
import gc
import torch
import torchvision
from utils import check_directory
from models.transfer_learning import get_transfer_learning_model
from attack.attack import attack

def attack_evaluation(target_model_config_dir,
                      attack_results_dir,
                      query_dir,
                      block_num=0,
                      query_budget=100,
                      effective_query_budget=100,
                      if_output_img=True,
		      attack_models=[
                        'resnet18',
                        'alexnet',
                        'alexnet_ptcv',
                        'densenet121',
                        'mobilenet_v2',
                        'vgg16',
                        'vgg19',
                      ],
		      attack_datasets = [
                        'VOCSegmentation',
                      ]
):

    torch.manual_seed(0)

    QUERY_CFG = {
        'query_img_dir': os.path.join(query_dir, str(query_budget)),
        'query_budget': query_budget
    }

    ATTACK_CFG = {
        'block_num': block_num,
        'iters': 30000
    }

    INPUT_SIZE = (3, 224, 224)
    MEAN=[0.485, 0.456, 0.406]
    STD=[0.229, 0.224, 0.225]

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(INPUT_SIZE[1:]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MEAN,
                                         std=STD)
    ])

    transform_MNIST = torchvision.transforms.Compose([
        torchvision.transforms.Resize(INPUT_SIZE[1:]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        torchvision.transforms.Normalize(mean=MEAN,
                                         std=STD)
    ])

    target_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(INPUT_SIZE[1:]),
        torchvision.transforms.ToTensor(),
    ])

    dataset_configs = {}
    dataset_configs['VOCSegmentation'] = {
        'kwargs':
            {
                'image_set': 'val',
                'year': '2012'
            },
        'transform': transform,
        'target_transform': target_transform
    }

    dataset_configs['NoiseDataset'] = {
        'kwargs':
        {
            'noise_size': INPUT_SIZE,
        },
        'transform': transform
    }

    dataset_configs['CelebADataset'] = {
        'kwargs':
        {
            'split': 'test',
        },
        'transform': transform
    }

    dataset_configs['MNIST'] = {
        'kwargs':
        {
            'train': False,
        },
        'transform': transform_MNIST
    }

    attack_config = {
        'mean': MEAN,
        'std': STD,
        'batch_size': 12 
    }



    def batch_attack(config_path,
                     synthetic_inputs_cached={},
                     probing_inputs_cached={}):
        attack_results = []
        # Log file
        log_sub_dir, attack_log_path_prefix = parse_config_path(config_path)
        attack_log_dir = os.path.join(attack_results_dir, log_sub_dir)
        check_directory(attack_log_dir)
        log_file_name = '[block_num_{}].json' \
                        .format(ATTACK_CFG['block_num'])
        attack_log_path = os.path.join(attack_results_dir, attack_log_path_prefix + log_file_name)
        if os.path.exists(attack_log_path):
            print(attack_log_path)
            return (synthetic_inputs_cached, probing_inputs_cached)

        target_model = get_transfer_learning_model(config_path).cpu()

        print("Batch Attack")
        for dataset_name in attack_datasets:
            for attack_model_name in attack_models:
                dataset_config = {
                    **{'dataset_name': dataset_name},
                    **dataset_configs[dataset_name],
                }
                print("Target model config path: {}".format(config_path))
                print("Attack model:{}".format(attack_model_name))
                print("Dataset {}".format(dataset_name))

                # Attack
                attack_info = {}
                query_config = {**QUERY_CFG,
                                **{'attack_model_name':attack_model_name}}
                if effective_query_budget < query_budget:
                    query_config['effective_query_budget'] = effective_query_budget

                print(len(synthetic_inputs_cached))

                (label_sim,
                 match_cnt,
                 support_cnt,
                 l1_distance,
                 cosine_similarity,
                 entropy,
                 synthetic_inputs_cached,
                 probing_inputs_cached) = attack(target_model,
                                                 query_config=query_config,
                                                 dataset_config=dataset_config,
                                                 attack_config={**ATTACK_CFG,
                                                                **attack_config},
                                                 if_output_img=if_output_img,
                                                 synthetic_inputs_cached=synthetic_inputs_cached,
                                                 probing_inputs_cached=probing_inputs_cached)

                attack_info['target_config_path'] = config_path
                attack_info['attack_model_name'] = attack_model_name
                attack_info['dataset_name'] = dataset_name
                attack_info['query_budget'] = query_config['query_budget']
                if 'effective_query_budget' in query_config.keys():
                    attack_info['effective_query_budget'] = query_config['effective_query_budget']
                attack_info['block_num'] = ATTACK_CFG['block_num']
                attack_info['label_similarity'] = label_sim
                attack_info['match_cnt'] = match_cnt
                attack_info['support_cnt'] = support_cnt
                attack_info['l1_distance'] = l1_distance
                attack_info['cosine_similarity'] = cosine_similarity
                attack_info['entropy'] = entropy
                attack_info['iters'] = ATTACK_CFG['iters']
                attack_results.append(attack_info)
                torch.cuda.empty_cache()

                with open(attack_log_path, 'w+') as f:
                    json.dump(attack_results, f)

        del target_model
        gc.collect()

        return (synthetic_inputs_cached, probing_inputs_cached)

    def parse_config_path(config_path):
        config_splits = config_path.split('/')
        needed_dir = ('/').join(config_splits[-4:-1])
        return needed_dir, ('/').join(config_splits[-4:]).split('.')[0]

    print('*'*20)
    synthetic_inputs_cached = {}
    probing_inputs_cached = {}
    for root, dirs, files in os.walk(target_model_config_dir):
        if len(files) == 0:
            continue
        if len(dirs) == 0:
            sub_config_dir = root
            for config_file in os.listdir(sub_config_dir):
                config_path = os.path.join(sub_config_dir, config_file)
                try:
                    synthetic_inputs_cached, probing_inputs_cached = batch_attack(config_path,
                                                                                  synthetic_inputs_cached,
                                                                                  probing_inputs_cached)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print()
                        print(str(e))
                        torch.cuda.empty_cache()
                        continue
        else:
            for sub_dir in dirs:
                sub_config_dir = os.path.join(root, sub_dir)
                for config_file in os.listdir(sub_config_dir):
                    config_path = os.path.join(sub_config_dir, config_file)

                    try:
                        (synthetic_inputs_cached,
                         probing_inputs_cached) = batch_attack(config_path,
                                                               synthetic_inputs_cached,
                                                               probing_inputs_cached)
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print()
                            print(str(e))
                            torch.cuda.empty_cache()
                            continue
