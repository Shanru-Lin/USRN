#!/bin/bash
#data="voc_1over32"
data=$1

## Train a supervised model with labelled images
python3 main.py --config configs/${data}_baseline.json

## Generate class-balanced subclass clusters
python3 main.py --save_feature True --resume saved/${data}_baseline/best_model.pth --config configs/${data}_baseline.json
python3 clustering.py --config configs/${data}_baseline.json --clustering_algorithm balanced_kmeans

## Train a semi-supervised model with both labelled and unlabelled images
python3 main.py --config configs/${data}_usrn.json

# python3 main.py --save_feature True --resume saved/voc_1over32_baseline/best_model.pth --config configs/voc_1over32_baseline.json
# python3 clustering.py --config configs/voc_1over32_baseline.json --clustering_algorithm normal_kmeans
# python3 clustering.py --config configs/voc_1over32_baseline.json --clustering_algorithm balanced_kmeans
# python3 main.py --config configs/voc_1over32_usrn.json


# testing usrn:
# python3 main.py --test True --resume saved/voc_1over32_usrn_parent/best_model.pth --config configs/test.json