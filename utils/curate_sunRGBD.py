# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os, shutil, torch
from glob import glob

import numpy as np

from PIL import Image

import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument("--dataroot", type=str, default="/home/czptc2h/datasets/SUNRGBD", help="image dir")
parser.add_argument(
    "--out_json_path", type=str, help="output jsonl path"
)
parser.add_argument(
    "--out_image_dir", type=str, help="output image folder"
)
args = parser.parse_args()


## restrict to 20 scenes
scene_dirs = glob(os.path.join(dataroot, "SUNRGBD/*/*/*"))
print("Scene Dirs:", scene_dirs)

out_json_path = args.out_json_path
out_image_path = args.out_image_dir

if os.path.exists(out_image_path):
    shutil.rmtree(out_image_path)
os.makedirs(out_image_path)

import shutil

points_per_image = 100
import json, os

count = 0
with open(out_json_path, "w") as jsonl_file:
    for scene_dir in scene_dirs:
        data_dict = {}
        ## Get image file path from scene directory
        print("Scene Dir:", scene_dir)
        try:
            image_path = glob(f"{scene_dir}/image/*")[0]
        except:
            image_path = glob(f"{scene_dir}/*/*/image/*")[0]
        if "NYU" in image_path:
            continue

        img = Image.open(image_path)
        sub_dir = image_path.replace(f"{dataroot}/SUNRGBD/", "")
        ## Copy the image to the out_image_path directory
        os.makedirs(os.path.dirname(out_image_path + "/" + sub_dir), exist_ok=True)
        shutil.copy(image_path, out_image_path + "/" + sub_dir)

        ## Get depth map file path from scene directory
        try:
            depth_path = glob(f"{scene_dir}/depth_bfx/*")[0]
        except:
            depth_path = glob(f"{scene_dir}/*/*/depth_bfx/*")[0]

        print("Image Path:", image_path, "; Depth Path:", depth_path)

        # Replace the last 2 file/folder names in the path of depth_path with "intrinsics.txt"
        intrinsic_path = os.path.join(os.path.dirname(os.path.dirname(depth_path)), "intrinsics.txt")

        with open(intrinsic_path, "r") as file:
            intrinsic_data = file.read().strip().split()
            intrinsic_matrix = np.array(intrinsic_data, dtype=np.float32).reshape(
                (3, 3)
            )
        print("Intrinsic Matrix:\n", intrinsic_matrix)

        # Read the image from image_path into a PIL image
        pil_image = Image.open(image_path)

        data_dict["image"] = sub_dir
        data_dict["intrinsics"] = [
            float(intrinsic_matrix[0, 0]),
            float(intrinsic_matrix[1, 1]),
            float(intrinsic_matrix[0, 2]),
            float(intrinsic_matrix[1, 2]),
        ] + [pil_image.size[0], pil_image.size[1]]

        depth_gt = Image.open(depth_path)
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = depth_gt / 10000.0

        # Randomly sample 100 pixels in depth_gt with value > 0.005 and < 25
        valid_pixels = np.argwhere((depth_gt > 0.005) & (depth_gt < 25))
        sampled_indices = np.random.choice(
            len(valid_pixels), size=points_per_image, replace=False
        )
        sampled_pixels = valid_pixels[sampled_indices]

        data_dict["pixel_coords"] = sampled_pixels[:, [1, 0]].tolist()
        fx, fy, cx, cy = (
            intrinsic_matrix[0, 0],
            intrinsic_matrix[1, 1],
            intrinsic_matrix[0, 2],
            intrinsic_matrix[1, 2],
        )
        z = depth_gt[sampled_pixels[:, 0], sampled_pixels[:, 1]]
        x = (sampled_pixels[:, 1] - cx) * z / fx
        y = (sampled_pixels[:, 0] - cy) * z / fy
        euclidean_distances = np.sqrt(x**2 + y**2 + z**2)
        data_dict["depth"] = euclidean_distances.tolist()

        print("PIL Image Size:", pil_image.size)
        print("Depth GT Size:", depth_gt.shape)

        print("Data Dictionary:", data_dict)

        json.dump(data_dict, jsonl_file)
        jsonl_file.write("\n")
        count += 1
        print(f"processed {count} images")
