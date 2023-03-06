# Teacher-Fingerprinting

This is the implementation of the attack proposed by the paper [Teacher Model Fingerprinting Attacks Against Transfer Learning](https://www.usenix.org/system/files/sec22-chen-yufei.pdf)

## Installtion
Clone this repo:
```bash
git clone https://github.com/yfchen1994/Teacher-Fingerprinting.git
cd Teacher_Fingerprinting
```

Then install dependencies by:
```bash
pip install -r requirements.txt
```

## Dataset
1. MNIST (part of TorchVision)
2. CIFAR-10 (part of TorchVision)
3. CIFAR-100 (part of TorchVision)
4. STL-10 (part of TorchVision)
5. VOC-Segmentation (part of TorchVision)
6. [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Please download `img_align_celeba` and `img_align_celeba.csv` from [this link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset). Then, create a subfolder `celeba` under `dataset`, and move the file `list_attr_celeba.csv` and the folder `img_align_celeba` into `celeba`.
7. [Dogs-vs-Cats](https://www.kaggle.com/c/dogs-vs-cats/data). Please download the original dataset, unzip `train.zip`, and move the folder `train` into the `./datasets/dogs_vs_cats/` folder. Then run the Python3 script `processdata.py` in `./datasets/dogs_vs_cats/`.

The structure of the `datasets` folder should be
```
datasets
+-- dogs_vs_cats 
|   +-- train
|   +-- data_info.csv
+-- celeba
|   +-- img_align_celeba
|   +-- list_attr_celeba.csv
```

## Usage
Please see the example `example.sh`.

## Citation
```
@inproceedings{CSWZ22,
author = {Yufei Chen and Chao Shen and Cong Wang and Yang Zhang},
title = {{Teacher Model Fingerprinting Attacks Against Transfer Learning}},
booktitle = {{USENIX Security Symposium (USENIX Security)}},
pages = {3593-3610},
publisher = {USENIX},
year = {2022}
}
```
