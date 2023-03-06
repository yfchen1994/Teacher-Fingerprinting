# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from utils import check_directory
import argparse

def generate_model_configs(config_root_dir,
                           model_root_dir,
                           model_name,
                           fcn_neuron,
                           dataset_config,
                           training_config,
                           pretrained=True,
                           fine_tuned_blocks=[],
                           selected_block_num=[]):
    """Generate configuration files of transfer learning models.

    Args:
        config_root_dir: the root directory of the model configuration files.
        model_root_dir:  the root directory of the model weights.
        model_name: the name of the teacher model. 
        fcn_neuron: the number of neurons for each fully connected layer.
                    E.g. [1024, 128, 10] means FCN(1024)->FCN(128)->FCN(10).
        dataset_config: the configuration of the student dataset.
        training_config: the configuration of transfer learning.
        pretrained: whether intilizing the convolution parts with pretrained weights. 
        fine_tuned_blocks: blocks to be fine-tuned. 
                           If set as an empty list, all the pretrained blocks are fixed.
        selected_block_num: how many pretrained blocks are used for transfer learning. 
                            Examples:  0 -- all the pretrained blocks selected.
                                      -1 -- all pretrained blocks except the last one selected. 
                                      -2 -- all pretrained blocks except the last two selected. 
                                       2 -- first two pretrained blocks selected. 
                            
    """
    yaml_content = {}
    dataset_name = dataset_config['dataset_name']

    # Create the directory to save the yaml files.
    yaml_file_dir = os.path.join(config_root_dir, dataset_name,
                                    model_name)
    check_directory(yaml_file_dir)
    # The exact path of the yaml file.
    neuron_str = ''
    for neuron_num in fcn_neuron:
        neuron_str += '_{}'.format(neuron_num)

    yaml_file_name = '{}{}.yaml'.format(model_name, neuron_str)
    yaml_file_path = os.path.join(yaml_file_dir, yaml_file_name)

    # Set up the yaml file content.
    yaml_content['model_dir'] = os.path.join(model_root_dir,
                                                dataset_name, model_name)

    yaml_content['model_config'] = {
        'input_size': (3, 224, 224),
        'pretrained_model': model_name,
        'fcn_neuron_num': fcn_neuron,
        'pretrained': pretrained
    }
    if dataset_name == 'CelebADataset':
    # Multi-label classification on CelebA.
        yaml_content['model_config']['is_multilabel_task'] = True

    if selected_block_num != 0:
        yaml_content['model_config']['selected_block_amount'] = selected_block_num

    if len(fine_tuned_blocks) > 0:
        yaml_content['model_config']['fine_tune_block_idx'] = fine_tuned_blocks

    yaml_content['dataset_config'] = dataset_config
    yaml_content['training_config'] = training_config

    with open(yaml_file_path, 'w') as yamlfile:
        yaml.dump(yaml_content, yamlfile, default_flow_style=False)

def set_configs(config_root_dir,
                model_root_dir,
                dataset_name='CIFAR100',
                model_name='alexnet',
                fcn_neuron=[128,10],
                epochs=20,
                batch_size=128,
                learning_rate=0.001,
                fine_tuned_blocks=[]):

    if dataset_name == 'CIFAR100':
        dataset_config = {
            'dataset_name': 'CIFAR100',
            'resized_size': (224, 224),
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'other_config': {
                'train': True
            }
        }
        output_shape = 100
    elif dataset_name == 'CIFAR10':
        dataset_config = {
            'dataset_name': 'CIFAR10',
            'resized_size': (224, 224),
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'other_config': {
                'train': True
            }
        }
        output_shape = 10
    elif dataset_name == 'MNIST':
        dataset_config = {
            'dataset_name': 'MNIST',
            'resized_size': (224, 224),
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'other_config': {
                'train': True
            }
        }
        output_shape = 10
    elif dataset_name == 'STL10':
        dataset_config = {
            'dataset_name': 'STL10',
            'resized_size': (224, 224),
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'other_config': {
                'split': 'train'
            }
        }
        output_shape = 10
    elif dataset_name == 'Dogs_vs_Cats':
        dataset_config = {
            'dataset_name': 'Dogs_vs_Cats',
            'resized_size': (224, 224),
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'other_config': {
                'split': 'train'
            }
        }
        output_shape = 2
    elif dataset_name == 'CelebADataset':
        dataset_config = {
            'dataset_name': 'CelebADataset',
            'resized_size': (224, 224),
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'other_config': {
                'split': 'train'
            }
        }
        output_shape = 40

    assert fcn_neuron[-1] == output_shape, \
    "The output shape of the classifier for dataset {} should be {}".format(dataset_name, output_shape)
    
    training_config = {
        'epochs': epochs, 
        'batch_size': batch_size, 
        'learning_rate': learning_rate
    }

    # Feel free to change where to save the model configurations 
    # and transfer learning models.
    generate_model_configs(config_root_dir,
                           model_root_dir,
                           model_name,
                           fcn_neuron,
                           dataset_config,
                           training_config,
                           pretrained=True,
                           fine_tuned_blocks=[],
                           selected_block_num=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate the configuration files for training the target models.")
    parser.add_argument('--config_root_dir', 
                        type=str,
                        help='Root directory to save the model configurations.')
    parser.add_argument('--model_root_dir', 
                        type=str,
                        help='Root directory to save the trained models.')
    parser.add_argument('--dataset_name', 
                        type=str,
                        default='CIFAR100', 
                        help='Dataset to train the student model.')
    parser.add_argument('--model_name', 
                        type=str,
                        default='alexnet', 
                        help='Teacher model used for transfer learning.')
    parser.add_argument('--fcn_neuron', 
                        type=int,
                        nargs='+',
                        default=[128,10], 
                        help='Teacher model used for transfer learning.')
    parser.add_argument('--epochs', 
                        type=int,
                        default=20, 
                        help='The "epochs" hyperparameter used for transfer learning.')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128,
                        help='The "batch size" hyperparameter used for transfer learning.')
    parser.add_argument('--learning_rate', 
                        type=float,
                        default=0.001, 
                        help='Learning rate used for transfer learning.')
    parser.add_argument('--fine_tuned_blocks', 
                        type=list,
                        default=[], 
                        help='Blocks to be fine tuned.')
    args = parser.parse_args()
    set_configs(config_root_dir=args.config_root_dir,
                model_root_dir=args.model_root_dir,
                dataset_name=args.dataset_name,
                model_name=args.model_name,
                fcn_neuron=args.fcn_neuron,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                fine_tuned_blocks=args.fine_tuned_blocks)


