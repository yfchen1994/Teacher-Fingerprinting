import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attack.attack_evaluation import attack_evaluation
import torch
import argparse

def launch_attack(target_model_config_dir,
                  attack_results_dir,
                  query_dir,
                  effective_query_budget=100,
                  block_num=0,
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
                  ]):
    """Launch attack.

    Args:
        target_model_config_dir: the location of the target models' configuration files. 
        attack_results_dir: where to save the attack results.
        query_dir: where the attack queries are.
        effective_query_budget: the query budget. (Note: #queries = 2*effective query budget)
        block_num: from which block to perform the attack. Defaults to 0.
        attack_models: candidate teacher models
        dataset_name
    """

    attack_evaluation(target_model_config_dir=target_model_config_dir,
                      attack_results_dir=attack_results_dir+'('+str(effective_query_budget)+')',
                      query_dir=query_dir,
                      block_num=block_num,
                      query_budget=100,
                      effective_query_budget=effective_query_budget,
                      attack_models=attack_models,
                      attack_datasets=attack_datasets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Launch the model fingerprinting attack.")
    parser.add_argument("--target_model_config_dir", 
                        type=str,
                        help="Directory storing the configuration files of the target models.")
    parser.add_argument("--attack_results_dir", 
                        type=str,
                        help="Directory to save the attack results.")
    parser.add_argument("--query_dir", 
                        type=str, 
                        help="Directory containing the attack queries.")
    parser.add_argument("--block_num", 
                        type=int,
                        default=0, 
                        help="From which block to generate the attack queries.")
    parser.add_argument("--effective_query_budget", 
                        type=int,
                        default=100, 
                        help="The query budget used by our attack.")
    parser.add_argument("--attack_models", 
                        type=str,
                        default=['alexnet','alexnet_ptcv'],
                        nargs='+',
                        help="Candidate teacher models.")
    parser.add_argument("--attack_datasets", 
                        type=str,
                        default=['VOCSegmentation'],
                        nargs='+', 
                        help="Probing datasets.")
    args = parser.parse_args()

    launch_attack(args.target_model_config_dir,
                  args.attack_results_dir,
                  args.query_dir,
                  args.effective_query_budget,
                  args.block_num,
                  args.attack_models,
                  args.attack_datasets)
