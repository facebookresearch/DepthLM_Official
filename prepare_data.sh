#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Though we provide separate code for each dataset, the main operation remains the same
# 1. download the dataset following the official instructions
# 2. convert images + camera intrinsics + depth maps into QA pairs, this step would generate a jsonl file containing all the meta data and a folder containing the corresponding images, similar to https://github.com/facebookresearch/DepthLM/tree/main/examples/ibims1.

# Argoverse
## 1. install av2 library and download + unzip the dataset following the official instructions in https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data
## 2. (optional) move 10-20 scenes from val to train folder to enlarge the training dataset size
## 3. curate the data
python utils/curate_argoverse.py \
"/path/to/argoverse/train_or_val/" \
"/path/to/output_image_folder" \
"/path/to/output_jsonl/argoverse2_train_or_val.jsonl"



# Waymo
## 1. download and unzip waymo open dataset from https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_2_0_1
## 2. curate the data
python utils/curate_waymo.py \
--dataset_dir /path/to/waymo/training/ \
--out_json_path /path/to/output_jsonl/waymo_train.jsonl \
--out_image_dir /path/to/output_image_folder

# NuScenes
## 1. download and unzip the dataset following https://www.nuscenes.org/nuscenes, we use the "Mini" subset for evaluation and other scenes in "All" for training
## 2. install nuscenes devkit at https://github.com/nutonomy/nuscenes-devkit
pip install nuscenes-devkit
## 3. curate training data
python utils/curate_nuscenes_train.py \
--dataroot /path/to/nuscenes_all \
--dataroot_mini /path/to/nuscenes_mini \
--out_json_path /path/to/output_jsonl/nuscenes_train.jsonl \
--out_image_dir /path/to/output_image_folder
## 4. curate eval data
python utils/curate_nuscenes_eval.py \
--dataroot /path/to/nuscenes_mini \
--out_json_path /path/to/output_jsonl/nuscenes_eval.jsonl \
--out_image_dir /path/to/output_image_folder

# ScanNet++
# our dataloader will automatically separate train and eval samples, so no need to separate them
## 1. download scannet++ dataset from https://kaldir.vc.in.tum.de/scannetpp/
## 2. clone and install the scannet++ github repo at https://github.com/scannetpp/scannetpp
## 3 change in /scannet_github_code_root/iphone/configs/prepare_iphone_data.yml the "data_doot" to the corresponding folder of your downloaded data
## 4. move data curation code to the scannet github local repo (we need modules in the scannet code to read the data)
mv utils/curate_scannet.py /scannet_github_code_root/iphone/prepare_depth_json.py
## 5. go to the scannet github local repo and run the data curation code
cd /scannet_github_code_root
python -m iphone.prepare_depth_json iphone/configs/prepare_iphone_data.yml

# Taskonomy
## 1. download the fullplus version of the dataset following https://github.com/StanfordVL/taskonomy/tree/master/data
## 2. curate data (coming soon)

# HM3d
## 1. download the hm3d dataset using https://docs.omnidata.vision/starter_dataset_download.html (set the components to hm3d)
## 2. curate data (coming soon)

# Matterport3D
## 1. download the dataset at https://niessner.github.io/Matterport/
## 2. curate data
python utils/curate_matterport3d.py \
--dataroot /path/to/matterport \
--out_json_path /path/to/output_jsonl/matterport.jsonl \
--out_image_dir /path/to/output_image_folder

# DDAD
## 1. download the dataset and install the dgp library following the "How to Use" section in https://github.com/TRI-ML/DDAD
## 2. curate data
python utils/curate_ddad.py \
--ddad_trainval_json_path /path/to/ddad/ddad_train_val/ddad.json \
--out_json_path /path/to/output_jsonl/ddad.jsonl \
--out_image_dir /path/to/output_image_folder  \
--path_to_dgp_lib /path/to/dgp/lib/folder

# ETH3D
## 1. download images and depth maps from https://www.eth3d.net/datasets
## 2. curate data
python utils/curate_eth3d.py \
--image_dir /path/to/eth3d/multi_view_training_dslr_jpg \
--depth_map_dir /path/to/eth3d/depth_map \
--out_json_path /path/to/output_jsonl/eth3d.jsonl \
--out_image_dir /path/to/output_image_folder

# sunRGBD & NYUv2
## 1. download data and unzip
dataroot=/path/to/sunRGBD
mkdir -p $dataroot
cd $dataroot
wget http://cvgl.stanford.edu/data2/sun_rgbd.tgz
tar -xvzf sun_rgbd.tgz
## 2. curate data for sunRGBD (without NYUv2)
python utils/curate_sunRGBD.py \
--dataroot /path/to/SUNRGBD/root \
--out_json_path /path/to/output_jsonl/sunRGBD.jsonl \
--out_image_dir /path/to/output_image_folder
## 3. curate data for NYUv2
python utils/curate_NYU.py \
--dataroot /path/to/SUNRGBD/root \
--out_json_path /path/to/output_jsonl/NYUv2.jsonl \
--out_image_dir /path/to/output_image_folder
