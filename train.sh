#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

model_path=$1
output_path=$2

# 1. we use ';' to separate the image_folder, dataset_name and sample_weights, as shown in this example, where we concat 2 identical datasets.
# 2. please follow the optimal hyper-paramters from the paper, this is just a basic example to make things run, for device that cannot use the same batch size as in the paper, you can scale the learning rate and batch size together following the square root rule so that you train with smaller batch sizes.
# 3. please adjust the max_steps to control the number of training samples
# 4. set per_device_train_batch_size to 10+ for H100 cards on Qwen2.5-VL 3B and 7B
# 5. to train the pixtral model, set per_device_train_batch_size to 3 and change the corresponding fsdp layer to --fsdp_transformer_layer_cls_to_wrap "MistralDecoderLayer,PixtralAttentionLayer"

torchrun --nproc_per_node=2 --master_port=12433 train.py \
--model_name_or_path $model_path \
--image_folder "./examples/ibims1/;./examples/ibims1/" \
--dataset_name "./examples/ibims1/ibims1_val.jsonl;./examples/ibims1/ibims1_val.jsonl" \
--sample_weights "1;1" \
--max_seq_length 4096 \
--learning_rate 1e-5 \
--lr_scheduler_type cosine \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 1 \
--warmup_ratio 0.1 \
--max_grad_norm 0.1 \
--logging_steps 1 \
--report_to tensorboard \
--gradient_checkpointing true \
--attn_implementation "flash_attention_2" \
--max_steps 10 \
--log_level info \
--logging_strategy steps \
--output_dir $output_path \
--save_steps 3000 \
--save_strategy steps \
--eval_strategy no \
--torch_dtype bfloat16 \
--seed 42 \
--normalized_focal_length 1000.0 \
--height_min 700 \
--height_max 1200 \
--width_min 1000 \
--width_max 1400 \
--dataset_class dataset_train \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap "Qwen2_5_VLDecoderLayer"
