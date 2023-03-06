# -*- coding: utf-8 -*-

import torchvision.models as models
import torch
from torch import nn
import types
import numpy as np
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import load_model

# In our experiment, the input size of models is 224x224.
INPUT_SIZE = [3, 224, 224]

class PretrainedModel(nn.Module):

    def __init__(self,
                 pretrained_model_name,
                 pretrained=True,
                 selected_block_amount=0):
        """Intializing the pretrained model.

        Args:
            pretrained_model_name: the name of the pretrained model, e.g. "alexnet." 
            pretrained: Whether the model comes from pretrained weights. Defaults to True.
            selected_block_amount: the amount selected blocks. Defaults to 0 (all selected).
        """
        super(PretrainedModel, self).__init__()
        # For convenience, we use lowercase characters for the model name.
        self.pretrained_name = pretrained_model_name.lower()
        self.pretrained = pretrained
        self.selected_block_amount = selected_block_amount
        self._prepare_pretrained_model()
        if 'backdoor' in self.pretrained_name:
            return

        module_dict = []
        block_lists = []
        self._BLOCK_STR = 'block'
        available_blocks_amount = len(self.block_split) - 1
        # Only use selected blocks.
        if self.selected_block_amount != 0:
            if self.selected_block_amount < 0:
                self.selected_block_amount = available_blocks_amount + self.selected_block_amount
            range_N = np.min(
                [available_blocks_amount, self.selected_block_amount])
            self.block_split = self.block_split[:range_N + 1]
        for i in range(len(self.block_split) - 1):
            block_name = self._BLOCK_STR + str(i)
            block = nn.Sequential(*list(self.pretrained_layers)
                                  [self.block_split[i]:self.block_split[i +
                                                                        1]])
            module_dict.append([block_name, block])
        self.pretrained_blocks = nn.ModuleDict(module_dict)
        print("Selected amount of blocks: {}".format(
            len(list(self.pretrained_blocks))))

    def forward(self, x):
        for block in self.pretrained_blocks:
            x = self.pretrained_blocks[block](x)
        return x

    def pretrained_forward(self, x, block_num=0):
        # Input -> teacher model -> Intermediate feature (at the BLOCK[block_num]).
        if block_num == 0:
            return self.forward(x)
        blocks = list(self.pretrained_blocks)
        max_N = len(blocks)
        if block_num < 0:
            block_num = max_N + block_num
        range_N = np.min([max_N, block_num])
        for i in range(range_N):
            x = self.pretrained_blocks[blocks[i]](x)
        return x

    def _prepare_pretrained_model(self):
        # Group the layers by blocks.
        # For more details, please refer to our paper:
        # https://arxiv.org/abs/2106.12478
        self.input_size = INPUT_SIZE

        if 'backdoored' in self.pretrained_name:
            self._prepare_model_backdoored()
        elif 'clean' in self.pretrained_name:
            self._prepare_model_clean()
        elif self.pretrained_name == 'alexnet':
            self._prepare_alexnet()
        elif self.pretrained_name == 'alexnet_ptcv':
            self._prepare_alexnet_ptcv()
        elif 'inception' in self.pretrained_name:
            self._prepare_inception()
        elif 'mobilenet' in self.pretrained_name:
            self._prepare_mobilenet()
        elif 'densenet' in self.pretrained_name:
            self._prepare_densenet()
        elif 'squeezenet' in self.pretrained_name:
            self._prepare_squeezenet()
        elif 'googlenet' in self.pretrained_name:
            self._prepare_googlenet()
        elif 'resnet' in self.pretrained_name:
            self._prepare_resnet()
        elif 'vgg' in self.pretrained_name:
            self._prepare_vgg()
        else:
            raise ValueError('Model {} not found!'.format(
                self.pretrained_name))

    def full_pretrained_model_forward(self, x):
        if self.pretrained_name == 'alexnet_ptcv':
            full_pretrained_model = ptcv_get_model("alexnetb",
                                                   pretrained=self.pretrained)
        else:
            full_pretrained_model = getattr(
                models, self.pretrained_name)(pretrained=self.pretrained)
        full_pretrained_model.eval()
        return full_pretrained_model(x.cpu())

    ### AlexNet
    def _prepare_alexnet(self):
        block_split = {}
        block_split[self.pretrained_name] = [0, 3, 6, 8, 10, 13]

        alexnet = getattr(models,
                          self.pretrained_name)(pretrained=self.pretrained)
        pretrained_layers = list(alexnet.features) + [alexnet.avgpool]
        self.pretrained_layers = pretrained_layers
        self.block_split = block_split[self.pretrained_name]

    ### AlexNet provided by PytorchCV
    ### (https://github.com/osmr/imgclsmob/blob/master/pytorch/README.md)
    def _prepare_alexnet_ptcv(self):
        block_split = {}
        block_split[self.pretrained_name] = [0, 2, 4, 5, 6, 8]
        alexnet_ptcv = ptcv_get_model("alexnetb", pretrained=self.pretrained)
        pretrained_layers = list(alexnet_ptcv.features.stage1) + \
                            list(alexnet_ptcv.features.stage2) + \
                            list(alexnet_ptcv.features.stage3) + \
                            [nn.AdaptiveAvgPool2d(output_size=(6, 6))]
        self.pretrained_layers = pretrained_layers
        self.block_split = block_split[self.pretrained_name]

    ### InceptionV3
    def _prepare_inception(self):
        block_split = {}
        block_split[self.pretrained_name] = [0] + [x + 1 for x in range(19)]

        inceptionv3 = getattr(models,
                              self.pretrained_name)(pretrained=self.pretrained)
        pretrained_layers = list(inceptionv3.children())[:-2]
        self.pretrained_layers = pretrained_layers
        self.block_split = block_split[self.pretrained_name]

    ### SqueezeNet
    def _prepare_squeezenet(self):
        squeezenet = getattr(models,
                             self.pretrained_name)(pretrained=self.pretrained)
        pretrained_layers = list(squeezenet.features)
        self.pretrained_layers = pretrained_layers

    ### VGG
    def _prepare_vgg(self):
        block_split = {}
        block_split['vgg16'] = [0, 5, 10, 17, 24, 33, 37]
        block_split['vgg19'] = [0, 5, 10, 19, 28, 39, 43]

        vgg = getattr(models, self.pretrained_name)(pretrained=self.pretrained)
        pretrained_layers = list(vgg.features) + \
                            [vgg.avgpool] + \
                            [nn.Flatten()] + \
                            list(vgg.classifier)[:4]
        self.pretrained_layers = pretrained_layers
        self.block_split = block_split[self.pretrained_name]

    ### DenseNet
    def _prepare_densenet(self):
        block_split = {}
        block_split[self.pretrained_name] = [0] + [x + 1 for x in range(12)]

        densenet = getattr(models,
                           self.pretrained_name)(pretrained=self.pretrained)
        self.pretrained_layers = list(densenet.features)
        self.block_split = block_split[self.pretrained_name]

    ### GoogleNet
    def _prepare_googlenet(self):
        block_split = {}
        block_split[self.pretrained_name] = [
            0, 2, 5, 6, 8, 9, 10, 11, 12, 14, 18
        ]

        googlenet = getattr(models,
                            self.pretrained_name)(pretrained=self.pretrained)
        self.pretrained_layers = list(googlenet.children())[:-1]
        self.block_split = block_split[self.pretrained_name]

    ### ResNet
    def _prepare_resnet(self):
        block_split = {}
        block_split['resnet18'] = [0, 4, 5, 6, 7, 9]

        resnet = getattr(models,
                         self.pretrained_name)(pretrained=self.pretrained)
        self.pretrained_layers = list(resnet.children())[:-1]

        self.block_split = block_split[self.pretrained_name]

    ### MobileNet
    def _prepare_mobilenet(self):
        block_split = {}
        block_split[self.pretrained_name] = [0] + [x + 1 for x in range(19)]

        mobilenet = getattr(models,
                            self.pretrained_name)(pretrained=self.pretrained)
        self.pretrained_layers = list(mobilenet.features)

        self.block_split = block_split[self.pretrained_name]

    ### Backdoored
    def _prepare_model_backdoored(self):
        model_path = './{}_infected_model.pkl'.format(
            self.pretrained_name).replace('_backdoored', '')

        if 'cifar10' in self.pretrained_name:
            from pretrained_model_inference.latent_backdoor.inject_latent_backdoor_cifar10 import get_init_model
        elif 'mnist' in self.pretrained_name:
            from pretrained_model_inference.latent_backdoor.inject_latent_backdoor_mnist import get_init_model
        elif 'stl10' in self.pretrained_name:
            from pretrained_model_inference.latent_backdoor.inject_latent_backdoor_stl10 import get_init_model
        else:
            raise ValueError('No such backdoored teacher: {}'.format(
                self.pretrained_name))

        model = get_init_model(
            self.pretrained_name.replace(
                self.pretrained_name.split('_')[0] + '_',
                '').replace('_backdoored', ''))
        model = load_model(model, model_path)
        self.pretrained_blocks = model.model_pretrain.pretrained_blocks
        self.classifier = model.classifier
