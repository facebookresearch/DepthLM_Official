#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

model_path=$1
python eval.py --model_path $model_path --image_folder "./examples/ibims1/" --json_path "./examples/ibims1/ibims1_val.jsonl" --bsz 3 --samples_to_eval 128
