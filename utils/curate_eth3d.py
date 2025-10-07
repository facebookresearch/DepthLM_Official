# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np

import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument("--image_dir", type=str, default="/home/czptc2h/datasets/ETH3D/multi_view_training_dslr_jpg", help="image dir")
parser.add_argument("--depth_map_dir", type=str, default="/home/czptc2h/datasets/ETH3D/depth", help="depth map dir")
parser.add_argument(
    "--out_json_path", type=str, help="output jsonl path"
)
parser.add_argument(
    "--out_image_dir", type=str, help="output image folder"
)
args = parser.parse_args()


def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
            ):
                image_paths.append(os.path.join(root, file))
    return image_paths


image_directory = args.image_dir
all_image_paths = get_image_paths(image_directory)
all_image_paths.sort(key=lambda x: os.path.basename(x))
print(all_image_paths[:10])

depth_directory = args.depth_map_dir
depth_image_paths = get_image_paths(depth_directory)
depth_image_paths.sort(key=lambda x: os.path.basename(x))
print(depth_image_paths[:10])


from PIL import Image

points_per_image = 100
out_json_path = args.out_json_path
out_image_path = args.out_image_dir
import shutil

if os.path.exists(out_image_path):
    shutil.rmtree(out_image_path)

os.makedirs(out_image_path)

import cv2


def undistort_fisheye(image, depth_image, camera_params):
    fx, fy, cx, cy = map(float, camera_params[4:8])
    k1, k2, p1, p2, k3, k4, sz1, sy1 = map(float, camera_params[8:])

    width, height = image.size
    k = np.array([k1, k2, k3, k4])
    p = np.array([p1, p2])
    sz = np.array([sz1, sy1])

    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Camera matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Distortion coefficients
    D = np.array([k1, k2, k3, k4])

    # Undistort image using OpenCV
    h, w = image_np.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )
    undistorted_image_np = cv2.remap(
        image_np,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # Convert back to PIL image
    undistorted_image = Image.fromarray(undistorted_image_np)

    # dont undistort depth image, get the original coordinate and use the mapping to get the undistorted coordinate
    # # Undistort depth image using OpenCV
    depth_image_np = np.array(depth_image)
    undistorted_depth_image_np = cv2.remap(
        depth_image_np,
        map1,
        map2,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    undistorted_depth_image = Image.fromarray(undistorted_depth_image_np)

    new_intrinsics = [fx, fy, cx, cy, width, height]
    return undistorted_image, undistorted_depth_image, new_intrinsics


import json

count = 0
with open(out_json_path, "w") as jsonl_file:
    for image_path, depth_path in zip(all_image_paths, depth_image_paths):
        image = Image.open(image_path)

        # image.save(os.path.join(out_image_path, "after_first_read.jpg"))
        with open(depth_path, "rb") as f:
            width, height = image.size
            depth_data = np.fromfile(f, dtype=np.float32, count=width * height)
            depth_image = depth_data.reshape((height, width))
        print(f"Loaded Image: {image_path}, Loaded Depth Map: {depth_path}")

        image_folder = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))
        dslr_calibration_folder = os.path.join(image_folder, "dslr_calibration_jpg")
        corresponding_camera_file = os.path.join(dslr_calibration_folder, "cameras.txt")
        if os.path.exists(corresponding_camera_file):
            print(
                f"Found corresponding camera file: {corresponding_camera_file} for image: {image_path}"
            )

        with open(corresponding_camera_file, "r") as camera_file:
            for line in camera_file:
                if not line.startswith("#"):
                    camera_params = line.strip().split(" ")
                    break
        print(f"Camera Parameters: {camera_params}")

        # Undistort image and depth_image
        fx, fy, cx, cy = map(float, camera_params[4:8])
        if camera_params[1] == "THIN_PRISM_FISHEYE":
            k1, k2, p1, p2, k3, k4, sz1, sy1 = map(float, camera_params[8:])

            # Call the function
            image, depth_image, new_intrinsics = undistort_fisheye(
                image, depth_image, camera_params
            )

        else:
            print("Camera model not supported")
            continue

        # Resize image and depth_image to have width < 2048
        scale_factor = min(1280.0 / height, 1)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height))
        depth_image = np.array(
            depth_image
        )  # dont rescale depth image, rescale the pixel_coordinates

        # Calculate new intrinsics
        fx *= scale_factor
        fy *= scale_factor
        cx *= scale_factor
        cy *= scale_factor
        new_intrinsics = [fx, fy, cx, cy, new_width, new_height]

        data_dict = {}
        data_dict["image"] = image_path.replace(
            image_directory+"/", ""
        ).replace(".png", ".jpg")
        data_dict["intrinsics"] = [
            float(new_intrinsics[0]),
            float(new_intrinsics[1]),
            float(new_intrinsics[2]),
            float(new_intrinsics[3]),
            int(new_intrinsics[4]),
            int(new_intrinsics[5]),
        ]

        data_dict["pixel_coords"] = []
        data_dict["depth"] = []

        valid_indices = np.argwhere(
            (depth_image > 1e-4)
            & (depth_image < 1e6)
            & (np.arange(depth_image.shape[0])[:, None] > 10)
            & (np.arange(depth_image.shape[0])[:, None] < depth_image.shape[0] - 10)
            & (np.arange(depth_image.shape[1])[None, :] > 10)
            & (np.arange(depth_image.shape[1])[None, :] < depth_image.shape[1] - 10)
        )

        sampled_indices = valid_indices[
            np.random.choice(valid_indices.shape[0], points_per_image, replace=False)
        ]

        for y, x in sampled_indices:
            x_ori = x
            y_ori = y
            x = int(x * scale_factor)
            y = int(y * scale_factor)
            data_dict["pixel_coords"].append([int(x), int(y)])
            fx, fy, cx, cy, width, height = data_dict["intrinsics"]
            x_normalized = (x - cx) / fx
            y_normalized = (y - cy) / fy
            z = float(depth_image[y_ori, x_ori])
            euclidean_distance = np.sqrt(x_normalized**2 + y_normalized**2 + 1) * z
            data_dict["depth"].append(float(euclidean_distance))

        # Save the resized image into out_image_path

        resized_image_path = os.path.join(out_image_path, data_dict["image"])
        print("resized_image_path", resized_image_path)

        os.makedirs(os.path.dirname(resized_image_path), exist_ok=True)
        image.save(resized_image_path)

        print("Data Dictionary:", data_dict)

        json.dump(data_dict, jsonl_file)
        jsonl_file.write("\n")
        count += 1
        print(f"processed {count} images")
