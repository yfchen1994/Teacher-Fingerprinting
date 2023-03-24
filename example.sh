#!/bin/bash

config_root_dir=./target_model_configs/fixed_features
attack_results_dir=./attack_results/fixed_features
effective_query_budget=100

# Step1: create transfer learning model configuration files 
python tools/generate_model_configs.py --config_root_dir $config_root_dir \
                                       --model_root_dir ./target_models/fixed_features \
                                       --dataset MNIST \
                                       --model_name alexnet \
                                       --fcn_neuron 128 10 \
                                       --epochs 5 \
                                       --batch_size 128 \
                                       --learning_rate 0.001
# Step2: train the target models
CUDA_VISIBLE_DEVICES=0 python tools/train_transfer_learning_model.py --config_root_dir $config_root_dir
# Step3: synthesize attack queries
CUDA_VISIBLE_DEVICES=0 python tools/launch_attack.py \
                              --target_model_config_dir $config_root_dir \
                              --attack_results_dir  $attack_results_dir\
                              --query_dir ./queries \
                              --effective_query_budget $effective_query_budget \
                              --attack_models alexnet resnet18 vgg16
# Step4: quering target model and get the attack results
python tools/check_attack_results.py \
       --attack_results_root_dir $attack_results_dir(\$effective_query_budget\) \
       --analysis_output attack_results.csv

