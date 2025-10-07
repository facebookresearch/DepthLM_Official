# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json, os

import cv2
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

import argparse
# Set up argument parser
parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument("--dataroot_mini", type=str, default="/home/czptc2h/datasets/nuscenes", help="data root mini")
parser.add_argument(
    "--out_json_path", type=str, help="output jsonl path"
)
parser.add_argument(
    "--out_image_dir", type=str, help="output image folder"
)
args = parser.parse_args()

# Initialize the NuScenes dataset
dataroot = args.dataroot_mini
version = "v1.0-trainval"  # Using mini version
nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)


def map_pointcloud_to_image(pointcloud, camera_token):
    """
    Map pointcloud to the image plane.

    Args:
        pointcloud: LidarPointCloud object
        camera_token: Token of the camera sample data

    Returns:
        points_img: Points in image coordinates
        depths: Depth values
    """
    cam = nusc.get("sample_data", camera_token)
    cam_path = os.path.join(nusc.dataroot, cam["filename"])
    im = cv2.imread(cam_path)

    # Get sensor calibration data
    lidar_to_world = nusc.get(
        "calibrated_sensor", pointcloud["calibrated_sensor_token"]
    )
    lidar_rotation = Quaternion(lidar_to_world["rotation"])
    lidar_translation = np.array(lidar_to_world["translation"])

    cam_to_world = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
    cam_intrinsic = np.array(cam_to_world["camera_intrinsic"])
    cam_rotation = Quaternion(cam_to_world["rotation"])
    cam_translation = np.array(cam_to_world["translation"])

    # Transform points from lidar to world coordinate
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, pointcloud["filename"]))
    points = pc.points[:3, :]
    points = np.vstack((points, np.ones(points.shape[1])))

    # Transformation matrix from lidar to world coordinate
    lidar_to_world_matrix = np.eye(4)
    lidar_to_world_matrix[:3, :3] = lidar_rotation.rotation_matrix
    lidar_to_world_matrix[:3, 3] = lidar_translation

    # Transformation matrix from world to camera coordinate
    world_to_cam_matrix = np.eye(4)
    world_to_cam_matrix[:3, :3] = cam_rotation.rotation_matrix.T
    world_to_cam_matrix[:3, 3] = -np.dot(
        cam_rotation.rotation_matrix.T, cam_translation
    )

    # Transform points to camera coordinate
    points_cam = np.dot(world_to_cam_matrix, np.dot(lidar_to_world_matrix, points))

    # Only keep points in front of the camera
    mask = points_cam[2, :] > 0
    points_cam = points_cam[:, mask]

    # Project to image plane
    points_img = np.dot(cam_intrinsic, points_cam[:3, :])
    points_img = points_img / points_img[2, :]
    points_img = points_img[:2, :]

    # Get depths
    depths = points_cam[2, :].copy()

    return points_img.T, depths, im


def create_depth_map(points_img, depths, image_shape):
    """
    Create a depth map from projected points.

    Args:
        points_img: Points in image coordinates
        depths: Depth values
        image_shape: Shape of the image (height, width)

    Returns:
        depth_map: Depth map as a 2D numpy array
    """
    depth_map = np.zeros((image_shape[0], image_shape[1]))

    # Keep only points that
    # Keep only points that fall within the image
    mask = np.logical_and.reduce(
        [
            points_img[:, 0] >= 0,
            points_img[:, 0] < image_shape[1],
            points_img[:, 1] >= 0,
            points_img[:, 1] < image_shape[0],
        ]
    )

    points_img = points_img[mask]
    depths = depths[mask]

    # Convert to integers for indexing
    points_int = np.floor(points_img).astype(np.int32)

    # Populate depth map
    for i in range(points_int.shape[0]):
        x, y = points_int[i, 0], points_int[i, 1]
        if depth_map[y, x] == 0 or depths[i] < depth_map[y, x]:
            depth_map[y, x] = depths[i]

    return depth_map


CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


def process_sample(sample_idx, output_folder, camera_name):
    """
    Process a single sample from the nuScenes dataset.

    Args:
        sample_idx: Index of the sample
        output_folder: Folder to save the image and depth map
    """
    data_dict = {}
    # Get sample
    sample = nusc.sample[sample_idx]

    # Get camera sample data
    camera_token = sample["data"][camera_name]
    # camera_data = nusc.get("sample_data", camera_token)

    # Get LiDAR sample data
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)

    # Map pointcloud to image and create depth map
    points_img, depths, image = map_pointcloud_to_image(lidar_data, camera_token)
    depth_map = create_depth_map(points_img, depths, (image.shape[0], image.shape[1]))
    # Read out camera intrinsic information
    cam_intrinsic = nusc.get(
        "calibrated_sensor",
        nusc.get("sample_data", camera_token)["calibrated_sensor_token"],
    )["camera_intrinsic"]
    print("Camera intrinsic matrix for", camera_name, ":", cam_intrinsic)
    # Print indices of the depth map that are not 0
    non_zero_indices = np.argwhere(depth_map != 0)
    print("Non-zero depth map indices:", non_zero_indices)
    # Print the size of the depth map and image
    print("Depth map size:", depth_map.shape)
    print("Image size:", image.shape)

    # Save image and depth map to output folder
    img_filename = f"{sample_idx:06d}_{camera_name}_image.jpg"
    cv2.imwrite(os.path.join(output_folder, img_filename), image)
    data_dict["image"] = img_filename
    data_dict["intrinsics"] = [
        cam_intrinsic[0][0],
        cam_intrinsic[1][1],
        cam_intrinsic[0][2],
        cam_intrinsic[1][2],
        image.shape[1],  # Image width
        image.shape[0],  # Image height
    ]
    # Randomly sample 100 pixels with non-zero depth map values
    non_zero_indices = np.argwhere(depth_map != 0)
    sampled_indices = non_zero_indices[
        np.random.choice(non_zero_indices.shape[0], 100, replace=False)
    ]

    # Store their pixel coordinates and depth values into lists
    data_dict["pixel_coords"] = [[int(x), int(y)] for y, x in sampled_indices]
    data_dict["depth"] = [depth_map[y, x] for y, x in sampled_indices]

    return data_dict


def process_multiple_samples(
    num_samples=5, output_folder="output", json_path="test.json", is_val=False
):
    """
    Process multiple samples from the dataset.

    Args:
        num_samples: Number of samples to process
        output_folder: Folder to save the images and depth maps
    """

    # Check if the output folder exists and delete it if it does
    if os.path.exists(output_folder):
        import shutil

        shutil.rmtree(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(json_path, "w") as f:
        if num_samples == -1:
            line_count = 0
            sample_range = (
                range(int(len(nusc.sample) * 0.95))
                if not is_val
                else range(int(len(nusc.sample) * 0.95), len(nusc.sample))
            )
            print("sample_range = ", sample_range)
            for i in sample_range:
                print(f"Processing sample {i}")
                for camera_name in CAMERA_NAMES:
                    entry = process_sample(i, output_folder, camera_name)
                    # Save meta_data_json to a JSON Lines file
                    json.dump(entry, f)
                    f.write("\n")
                    line_count += 1
            print(f"Total lines processed: {line_count}")
        else:
            for i in np.random.choice(
                len(nusc.sample), min(num_samples, len(nusc.sample)), replace=False
            ):
                print(f"Processing sample {i}")
                camera_name = np.random.choice(CAMERA_NAMES)
                entry = process_sample(i, output_folder, camera_name)
                # Save meta_data_json to a JSON Lines file
                json.dump(entry, f)
                f.write("\n")


# Example: Process all samples and save to "output" folder
process_multiple_samples(
    num_samples=-1,
    output_folder=args.out_image_dir,
    json_path=args.out_json_path,
    is_val=True,
)
