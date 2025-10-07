# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json, random
import os
import sys

import numpy as np
import torch
from PIL import Image

import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument("--dataroot", type=str, default="/home/czptc2h/datasets/matterport", help="data root")
parser.add_argument(
    "--out_json_path", type=str, help="output jsonl path"
)
parser.add_argument(
    "--out_image_dir", type=str, help="output image folder"
)
args = parser.parse_args()

root = args.dataroot

def get_all_image_paths(root):
    image_paths = []
    for subdir, _, files in os.walk(root):
        if "undistorted_color_images" in subdir:
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")):
                    image_paths.append(os.path.join(subdir, file))
    return image_paths


def get_all_file_paths(root, folder_name, file_extensions=(".png")):
    image_paths = []
    for subdir, _, files in os.walk(root):
        if folder_name in subdir:
            for file in files:
                if file.endswith(file_extensions):
                    image_paths.append(os.path.join(subdir, file))
    return image_paths


all_image_paths = sorted(get_all_image_paths(root))
all_depth_paths = sorted(get_all_file_paths(root, "undistorted_depth_images", (".png")))
all_calib_paths = sorted(
    get_all_file_paths(root, "undistorted_camera_parameters", (".conf"))
)

calib_map_dict = {}
for calib_path in all_calib_paths:
    folder_name = os.path.relpath(calib_path, root).split(os.sep)[0]
    calib_map_dict[folder_name] = calib_path

points_per_image = 100
out_json_path = args.out_json_path
# Create the directory for out_json_path if it doesn't exist
os.makedirs(os.path.dirname(out_json_path), exist_ok=True)

out_image_path = args.out_image_dir
import shutil

if os.path.exists(out_image_path):
    shutil.rmtree(out_image_path)
os.makedirs(out_image_path)

count = 0
with open(out_json_path, "w") as jsonl_file:
    for image_path, depth_path in zip(all_image_paths, all_depth_paths):
        folder_name = os.path.relpath(image_path, root).split(os.sep)[0]
        calib_path = calib_map_dict[folder_name]
        # Extract the base filename from the image_path
        base_filename = os.path.basename(image_path)

        # Initialize variables to store the intrinsics matrix
        intrinsics_matrix = None

        # Read the calibration file
        with open(calib_path, "r") as calib_file:
            for line in calib_file:
                # Check if the line contains the base filename
                if base_filename in line:
                    # Read the previous line for intrinsics_matrix
                    calib_file.seek(0)  # Reset file pointer to the beginning
                    lines = calib_file.readlines()
                    for i, l in enumerate(lines):
                        if base_filename in l:
                            # The intrinsics_matrix is expected to be in the lines before the scan line
                            for j in range(i - 1, -1, -1):
                                intrinsics_matrix_line = lines[j]
                                if "intrinsics_matrix" in intrinsics_matrix_line:
                                    # Extract the values after 'intrinsics_matrix'
                                    intrinsics_matrix = list(
                                        map(float, intrinsics_matrix_line.split()[1:])
                                    )
                                    break
                    break

        if not intrinsics_matrix:
            print(f"Intrinsics Matrix not found for {base_filename}")
            continue

        data_dict = {}
        data_dict["image_path"] = folder_name + "/" + base_filename

        # Read the image at image_path as a PIL image
        pil_image = Image.open(image_path)
        fx = intrinsics_matrix[0]
        fy = intrinsics_matrix[4]
        cx = intrinsics_matrix[2]
        cy = intrinsics_matrix[5]

        data_dict["intrinsics"] = [
            fx,
            fy,
            cx,
            cy,
            pil_image.width,
            pil_image.height,
        ]
        # Read the depth image at depth_path as a PIL image
        depth_pil_image = Image.open(depth_path)

        # Convert depth image to numpy array for easier manipulation
        depth_array = np.array(depth_pil_image)

        # Get the coordinates where depth is not 0
        non_zero_coords = np.argwhere(depth_array > 0)

        # Randomly sample 2 * points_per_image coordinates
        sample_size = min(len(non_zero_coords), 2 * points_per_image)
        if sample_size < 50:
            continue
        if len(non_zero_coords) < 2 * points_per_image:
            print(
                f"Population size: {len(non_zero_coords)} is smaller than required sample size: {2 * points_per_image}"
            )

        sampled_coords = random.sample(list(non_zero_coords), sample_size)

        # Extract intrinsics from data_dict
        fx, fy, cx, cy, width, height = data_dict["intrinsics"]

        # Initialize a list to store the 3D points
        euclidean_distances = []

        # Iterate over the sampled coordinates
        for coord in sampled_coords:
            y, x = coord
            # Get the depth value at the sampled coordinate
            depth = depth_array[y, x] / 4000.0

            x_real = (x - cx) * depth / fx
            y_real = (y - cy) * depth / fy
            z_real = depth
            # Calculate the Euclidean distance
            euclidean_distances.append(
                float(np.sqrt(x_real**2 + y_real**2 + z_real**2))
            )

        # Filter and collect the first points_per_image elements that satisfy the conditions
        filtered_coords = []
        filtered_distances = []

        for coord, distance in zip(sampled_coords, euclidean_distances):
            if 0.05 <= distance <= 50:
                filtered_coords.append([int(coord[1]), int(coord[0])])  # [x, y] format
                filtered_distances.append(distance)
                if len(filtered_coords) == points_per_image:
                    break

        # Set the data_dict values
        data_dict["pixel_coords"] = filtered_coords
        data_dict["depth"] = filtered_distances

        # Save the pil_image to the relative path of data_dict["image_path"] under out_image_path
        relative_image_path = os.path.join(out_image_path, data_dict["image_path"])
        os.makedirs(os.path.dirname(relative_image_path), exist_ok=True)
        pil_image.save(relative_image_path)

        # Write data_dict into jsonl_file
        jsonl_file.write(json.dumps(data_dict) + "\n")

        count += 1
        if count % 1000 == 0:
            print(f"Iteration: {count}")
            print(f"Data Dictionary: {data_dict}")
