# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument(
    "--out_json_path", type=str, help="output jsonl path"
)
parser.add_argument(
    "--out_image_dir", type=str, help="output image folder"
)
parser.add_argument(
    "--ddad_trainval_json_path", type=str, help="path to the ddad train val json path, i.e., ddad/ddad_train_val/ddad.json"
)
parser.add_argument(
    "--path_to_dgp_lib", type=str, help="dgp path"
)
args = parser.parse_args()

sys.path.insert(
    0,
    args.path_to_dgp_lib,
)  # add nuscenes package path to enable module finding

import cv2
import numpy as np
import PIL

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.proto.ontology_pb2 import Ontology
from dgp.utils.protobuf import open_pbobject
from dgp.utils.visualization_utils import visualize_semantic_segmentation_2d

# from IPython import display
from matplotlib.cm import get_cmap


plasma_color_map = get_cmap("plasma")


out_json_path = args.out_json_path
output_image_path = args.out_image_dir
points_per_image = 100

import os

# Remove the folder of output_image_path if it exists
if os.path.exists(output_image_path):
    import shutil

    shutil.rmtree(output_image_path)

# Ensure the output directory exists
os.makedirs(output_image_path, exist_ok=True)

# Define high level variables
DDAD_TRAIN_VAL_JSON_PATH = args.ddad_trainval_json_path
DATUMS = ["lidar"] + ["CAMERA_%02d" % idx for idx in [1, 5, 6, 7, 8, 9]]

# Load the val set
ddad_val = SynchronizedSceneDataset(
    DDAD_TRAIN_VAL_JSON_PATH,
    split="val",
    datum_names=DATUMS,
    generate_depth_from_datum="lidar",
)
print("Loaded DDAD val split containing {} samples".format(len(ddad_val)))

import json

# Open the out_json_path as a jsonl file for writing
with open(out_json_path, "w") as jsonl_file:
    count = 0
    # Iterate through the dataset.
    for sample in ddad_val:
        # Each sample contains a list of the requested datums.
        print("sample = {}", sample, "/", len(sample))

        for i in range(len(sample[0])):
            datum = sample[0][i]
            if "CAMERA" in datum["datum_name"]:
                data_dict = {}
                image_fname = f"{count}.jpg"
                data_dict["image"] = f"val_images/" + image_fname

                print(datum["datum_name"], i)
                # point_cloud = lidar["point_cloud"]  # Nx3 numpy.ndarray
                image_01 = datum["rgb"]  # PIL.Image
                depth_01 = datum["depth"]  # (H,W) numpy.ndarray, generated from 'lidar'

                data_dict["intrinsics"] = [
                    float(datum["intrinsics"][0, 0]),
                    float(datum["intrinsics"][1, 1]),
                    float(datum["intrinsics"][0, 2]),
                    float(datum["intrinsics"][1, 2]),
                    image_01.size[0],  # Image width
                    image_01.size[1],  # Image height
                ]

                # print("image_01 = ", image_01, "; depth_01 = ", depth_01)
                # Find non-zero elements in depth_01
                non_zero_indices = np.nonzero(depth_01)
                random_indices = np.random.choice(
                    len(non_zero_indices[0]), size=100, replace=False
                )
                non_zero_indices = (
                    non_zero_indices[0][random_indices],
                    non_zero_indices[1][random_indices],
                )
                non_zero_values = depth_01[non_zero_indices]

                data_dict["pixel_coords"] = []
                data_dict["depth"] = []
                # Print pixel coordinates and their corresponding depth values
                for coord, value in zip(zip(*non_zero_indices), non_zero_values):
                    data_dict["pixel_coords"].append([int(coord[1]), int(coord[0])])
                    data_dict["depth"].append(float(value))
                    # print(f"Pixel coordinates: {coord}, Depth value: {value}")

                # # Calculate and print the minimum and maximum values in the non-zero depth values
                # min_depth = np.min(non_zero_values)
                # max_depth = np.max(non_zero_values)
                # print(
                #     f"Minimum depth value: {min_depth}, Maximum depth value: {max_depth}"
                # )
                print("data_dict = ", data_dict)
                json.dump(data_dict, jsonl_file)
                jsonl_file.write("\n")

                # Save image_01 to the specified path
                image_save_path = os.path.join(output_image_path, image_fname)
                image_01.save(image_save_path)
                count += 1
                print(f"processed {count} images")

                # breakpoint()
                # break
