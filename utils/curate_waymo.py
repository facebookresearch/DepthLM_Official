# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io, warnings
from typing import Optional

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action="ignore", category=FutureWarning)


import argparse

# Print the pixel coordinates and their depth values
import random

import dask.dataframe as dd
import numpy as np
import tensorflow as tf
from PIL import Image
from waymo_open_dataset import v2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.v2.perception.utils import lidar_utils

import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument("--dataset_dir", type=str, default="/home/czptc2h/datasets/waymo/training/", help="waymo")
parser.add_argument(
    "--out_json_path", type=str, help="output jsonl path"
)
parser.add_argument(
    "--out_image_dir", type=str, help="output image folder"
)
args = parser.parse_args()

# Path to the directory with all components
dataset_dir = args.dataset_dir
# List all parquet files in the "camera_image" directory and extract their names without extensions
camera_image_dir = f"{dataset_dir}/camera_image"
import os

parquet_files = [
    os.path.join(camera_image_dir, file)
    for file in os.listdir(camera_image_dir)
    if file.endswith(".parquet")
]
filenames = [
    os.path.splitext(file)[0]
    for file in os.listdir(camera_image_dir)
    if file.endswith(".parquet")
]

# change these to process a subset of the data
start_file = 0
end_file = len(filenames)

print(f"Found {len(filenames)} files in {camera_image_dir}")
# Print some samples of filenames
sample_size = min(
    5, len(filenames)
)  # Print up to 5 samples or less if fewer files exist
print("Sample filenames:", filenames[:sample_size])


def read(tag: str, context_name: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = tf.io.gfile.glob(f"{dataset_dir}/{tag}/{context_name}.parquet")
    return dd.read_parquet(paths)


out_json_path = args.out_json_path
# Create the directory for out_json_path if it doesn't exist
os.makedirs(os.path.dirname(out_json_path), exist_ok=True)

out_image_path = args.out_image_dir

import shutil

if os.path.exists(out_image_path):
    shutil.rmtree(out_image_path)
os.makedirs(out_image_path)

points_per_image = 100
import json, os

import cv2

import numpy as np
from PIL import Image


def undistort_image(pil_image, intrinsic, pixel_coordinates):
    # Convert PIL image to OpenCV image
    cv_image = np.array(pil_image)
    # Define the camera intrinsic parameters
    fx = intrinsic.f_u
    fy = intrinsic.f_v
    cx = intrinsic.c_u
    cy = intrinsic.c_v
    k1 = intrinsic.k1
    k2 = intrinsic.k2
    p1 = intrinsic.p1
    p2 = intrinsic.p2
    k3 = intrinsic.k3
    # Create a camera intrinsic matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # Create a distortion coefficients vector
    dist_coeffs = np.array([k1, k2, p1, p2, k3])
    # Get the image dimensions
    h, w = cv_image.shape[:2]
    # Create a new camera intrinsic matrix with the distortion removed
    # Create a new camera intrinsic matrix with the distortion removed
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    new_K[0, 0] = fx  # Set fx
    new_K[1, 1] = fy  # Set fy
    new_K[0, 2] = w / 2  # Set cx to be at the center
    new_K[1, 2] = h / 2  # Set cy to be at the center
    # Undistort the image
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), 5)
    undistorted_image = cv2.remap(cv_image, map_x, map_y, cv2.INTER_LINEAR)
    # Convert the undistorted image back to a PIL image
    undistorted_pil_image = Image.fromarray(undistorted_image)

    # Convert pixel coordinates to undistorted coordinates
    undistorted_pixel_coordinates = []
    for x, y in pixel_coordinates:
        undistorted_x = int(map_x[y, x])
        undistorted_y = int(map_y[y, x])
        undistorted_pixel_coordinates.append((undistorted_x, undistorted_y))

    return undistorted_pil_image, new_K, undistorted_pixel_coordinates


points_per_image = 100
count = 0
with open(out_json_path, "w") as jsonl_file:
    for filename in filenames[start_file:end_file]:
        # Process each filename as needed
        # Example: Write filename to the JSONL file
        # print("Processing filename:", filename)
        lidar = read("lidar", filename)
        lidar_calib = read("lidar_calibration", filename)
        camera_calib = read("camera_calibration", filename)
        lidar_pose = read("lidar_pose", filename)
        vehicle_pose = read("vehicle_pose", filename)
        cam_img = read("camera_image", filename)
        lidar_camera_projection = read("lidar_camera_projection", filename)
        df = v2.merge(lidar_calib, lidar)
        df = v2.merge(df, lidar_camera_projection)
        df = v2.merge(df, lidar_pose)
        df = v2.merge(df, vehicle_pose)
        df = v2.merge(df, camera_calib)
        df = v2.merge(df, cam_img)

        for _, row in df.iterrows():
            # print(row)

            # Create all component objects
            lidar = v2.LiDARComponent.from_dict(row)
            lidar_calib = v2.LiDARCalibrationComponent.from_dict(row)
            camera_calib = v2.CameraCalibrationComponent.from_dict(row)
            lidar_pose = v2.LiDARPoseComponent.from_dict(row)
            vehicle_pose = v2.VehiclePoseComponent.from_dict(row)
            camera_image = v2.CameraImageComponent.from_dict(row)
            lidar_cam_proj = v2.LiDARCameraProjectionComponent.from_dict(row)

            range_image_cartesian = lidar_utils.convert_range_image_to_cartesian(
                range_image=lidar.range_image_return1,
                calibration=lidar_calib,
                pixel_pose=lidar_pose.range_image_return1,
                frame_pose=vehicle_pose,
            )
            extrinsic = np.reshape(camera_calib.extrinsic.transform, [1, 4, 4]).astype(
                np.float32
            )
            camera_image_size = (camera_calib.height, camera_calib.width)
            ric_shape = range_image_cartesian.shape
            ric = np.reshape(
                range_image_cartesian, [1, ric_shape[0], ric_shape[1], ric_shape[2]]
            )

            cp = lidar_cam_proj.range_image_return1
            cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.values), cp.shape)
            cp_shape = cp_tensor.shape
            cp_tensor = np.reshape(
                cp_tensor, [1, cp_shape[0], cp_shape[1], cp_shape[2]]
            )

            depth_image = range_image_utils.build_camera_depth_image(
                ric,
                extrinsic,
                cp_tensor,
                list(camera_image_size),
                camera_image.key.camera_name,
            )

            # Convert depth_image to a numpy array
            depth_image_np = depth_image.numpy().squeeze(axis=0)

            # Find non-zero elements in the depth_images
            non_zero_indices = np.nonzero(depth_image_np)

            # Extract the pixel coordinates and their corresponding depth values
            pixel_coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
            # breakpoint()
            depth_values = depth_image_np[non_zero_indices]

            data_dict = {}
            data_dict["image"] = (
                f"{camera_image.key.segment_context_name}/{camera_image.key.frame_timestamp_micros}_{camera_image.key.camera_name}.jpg"
            )

            sample_size = min(2 * points_per_image, len(pixel_coordinates))
            sample_indices = random.sample(range(len(pixel_coordinates)), sample_size)

            data_dict["pixel_coords"] = [
                list(reversed(pixel_coordinates[i])) for i in sample_indices
            ]

            data_dict["depth"] = [float(depth_values[i]) for i in sample_indices]

            image_filename = os.path.join(
                out_image_path,
                data_dict["image"],
            )
            pil_image = Image.open(io.BytesIO(camera_image.image))

            undistorted_pil_image, new_K, data_dict["pixel_coords"] = undistort_image(
                pil_image, camera_calib.intrinsic, data_dict["pixel_coords"]
            )

            # Check if the fx value in new_K is greater than 1000
            if new_K[0, 0] > 1000:
                # Calculate the scaling factor to make fx equal to 1000
                scale_factor = 1000 / new_K[0, 0]

                # Rescale the undistorted_pil_image
                new_width = int(undistorted_pil_image.width * scale_factor)
                new_height = int(undistorted_pil_image.height * scale_factor)
                undistorted_pil_image = undistorted_pil_image.resize(
                    (new_width, new_height), Image.ANTIALIAS
                )

                # Rescale the new_K matrix
                new_K[0, 0] *= scale_factor
                new_K[1, 1] *= scale_factor
                new_K[0, 2] *= scale_factor
                new_K[1, 2] *= scale_factor

                # Rescale the pixel coordinates
                data_dict["pixel_coords"] = [
                    (int(x * scale_factor), int(y * scale_factor))
                    for x, y in data_dict["pixel_coords"]
                ]

            data_dict["intrinsics"] = [
                new_K[0, 0],
                new_K[1, 1],
                new_K[0, 2],
                new_K[1, 2],
                undistorted_pil_image.width,
                undistorted_pil_image.height,
            ]

            # Filter pixel coordinates and corresponding depth values
            valid_pixel_coords = []
            valid_depths = []
            for coord, depth in zip(data_dict["pixel_coords"], data_dict["depth"]):
                x, y = coord
                if (
                    0 <= x < undistorted_pil_image.width
                    and 0 <= y < undistorted_pil_image.height
                ):
                    valid_pixel_coords.append(coord)
                    valid_depths.append(depth)
                if len(valid_pixel_coords) == points_per_image:
                    break

            data_dict["pixel_coords"] = valid_pixel_coords
            data_dict["depth"] = valid_depths

            os.makedirs(os.path.dirname(image_filename), exist_ok=True)
            undistorted_pil_image.save(image_filename)

            json.dump(data_dict, jsonl_file)
            jsonl_file.write("\n")

            if count % 100 == 0:
                print(f"data_dict[{count}] = ", data_dict)

            count += 1
        count += 1
