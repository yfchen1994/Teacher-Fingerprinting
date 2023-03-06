# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np


class TransferLearningModel(nn.Module):

    def __init__(self,
                 model_pretrain,
                 fcn_neuron_num,
                 input_size,
                 is_multilabel_task=False,
                 fine_tune_block_idx=[]):
        """Initializing the transfer learning model.

        Args:
            model_pretrain: the pretrained teacher model.
            fcn_neuron_num: the number of neurons for each layer.
                            E.g. [1024, 128, 10] means FCN(1024)->FCN(128)->FCN(10).
            input_size: the input size of the model.
            is_multilabel_task: whether this is a multilabel classification model. 
                                Defaults to False.
            fine_tune_block_idx: indices of blocks that are finetuned. Defaults to [].
        """
        super(TransferLearningModel, self).__init__()
        # Fix the parameters of the pretrained part.
        self.model_pretrain = model_pretrain.cuda()
        self.input_size = input_size
        self.is_multilabel_task = is_multilabel_task
        self.fine_tune_block_idx = fine_tune_block_idx

        # Store the input size for each fcn layer.
        x = torch.zeros((1, ) + self.input_size).cuda()

        classifier_layers = []
        if 'squeezenet' in model_pretrain.pretrained_name:
            classifier_layers = [
                nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
        classifier_layers = classifier_layers + [nn.Flatten()]

        self.fcn_in_size = self.model_pretrain(x) \
                           .data.view(1, -1).size(1)
        self.fcn_neuron_num = [self.fcn_in_size] + fcn_neuron_num

        classifier_layer_num = len(self.fcn_neuron_num)
        for i in range(classifier_layer_num - 1):
            classifier_layers.append(
                nn.Linear(self.fcn_neuron_num[i], self.fcn_neuron_num[i + 1]))
            if i < classifier_layer_num - 2:
                classifier_layers.append(
                    nn.BatchNorm1d(self.fcn_neuron_num[i + 1]))
                classifier_layers.append(nn.ReLU())
                classifier_layers.append(nn.Dropout(0.5))
        if self.is_multilabel_task:
            classifier_layers.append(nn.Sigmoid())
        else:
            classifier_layers.append(nn.Softmax(dim=1))

        # Register fcn layers to the model graph.
        self.classifier = nn.Sequential(*classifier_layers)
        self.setup_trainable_params()

    def setup_trainable_params(self):
        blocks = list(self.model_pretrain.pretrained_blocks)

        def freeze_bn(block):
            # Freezing the batch_normalization layer.
            for child in block.children():
                if len(list(child.children())) > 0:
                    freeze_bn(child)
                else:
                    if isinstance(child, nn.BatchNorm2d):
                        #child.track_running_stats = False
                        child.eval()

        def unfreeze_bn(block):
            # Unfreezing the batch_normalization layer.
            for child in block.children():
                if len(list(child.children())) > 0:
                    unfreeze_bn(child)
                else:
                    if isinstance(child, nn.BatchNorm2d):
                        #child.track_running_stats = True
                        child.train()

        if self.model_pretrain.pretrained:
            # Freeze the batch normalization layer
            for block in blocks:
                freeze_bn(self.model_pretrain.pretrained_blocks[block])

            for param in self.model_pretrain.parameters():
                param.requires_grad = False
                #print("{}: Freeze BN".format(self.model_pretrain.pretrained_name))

            for i in self.fine_tune_block_idx:
                print("Unfreeze block {}".format(i))
                for param in self.model_pretrain.pretrained_blocks[
                        blocks[i]].parameters():
                    param.requires_grad = True
                unfreeze_bn(self.model_pretrain.pretrained_blocks[blocks[i]])
                #print("{}: Unfreeze BN".format(self.model_pretrain.pretrained_name))
        else:
            for param in self.model_pretrain.parameters():
                param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model_pretrain(x)
        #for fcn_layer in self.fcn_layers:
        x = self.classifier(x)
        return x

    def pretrained_forward(self, x, block_num=0):
        # Input -> [teacher model] -> Intermediate feature.
        if block_num == 0:
            return self.model_pretrain(x)
        blocks = list(self.model_pretrain.pretrained_blocks)
        max_N = len(blocks)
        if block_num < 0:
            block_num = max_N + block_num
        range_N = np.min([max_N, block_num])
        for i in range(range_N):
            x = self.model_pretrain.pretrained_blocks[blocks[i]](x)
        return x

    def fcn_forward(self, conv_feature):
        # Intermediate feature -> [student model] -> Output.
        x = conv_feature
        x = self.fcn_layers(x)
        return x
