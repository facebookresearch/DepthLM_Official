# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import os
import sys
from pathlib import Path
from typing import Final

import av2.rendering.color as color_utils
import av2.rendering.rasterize as raster_rendering_utils
import av2.rendering.video as video_utils
import av2.utils.io as io_utils
import av2.utils.raster as raster_utils

import click
import cv2
import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.color import GREEN_HEX, RED_HEX
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt
from numpy import random
from PIL import Image

logger = logging.getLogger(__name__)


NUM_RANGE_BINS: Final[int] = 50
RING_CAMERA_FPS: Final[int] = 20


def get_immediate_subfolders(folder_path: str) -> list:
    """Return a list of immediate subfolders in the given folder path."""
    return [f.name for f in Path(folder_path).iterdir() if f.is_dir()]


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python script.py <root_folder> <out_image_folder> <jsonl_output_path>"
        )
        sys.exit(1)

    root_folder = sys.argv[1]
    out_image_folder = sys.argv[2]
    jsonl_output_path = sys.argv[3]

    frame_sample_interval = 1
    points_per_frame = 100 # by default we curate 100 labeled pixels per image which is more than enough for depth estimation, you can change this number to have more curated pixels
    cameras_used = [
        "ring_front_left",
        "ring_front_right",
        "ring_rear_left",
        "ring_rear_right",
        "ring_side_left",
        "ring_side_right",
        "ring_front_center",
        "stereo_front_left",
        "stereo_front_right",
    ]
    folders = get_immediate_subfolders(root_folder)
    print(f"there are {len(folders)} folders in total, the first one is {folders[0]}")

    loader = AV2SensorDataLoader(
        data_dir=Path(root_folder), labels_dir=Path(root_folder)
    )

    count = 0
    count_rows = 0
    with open(jsonl_output_path, "w") as f:
        skip_log_id = "d37be0e2-8223-3eeb-a0e2-c4b75d5ff87b"  # errors during my downloading for this log, comment if you dont have issues
        skip = False
        for log_id in folders[:2]:
            if skip and log_id != skip_log_id:
                continue
            skip = False
            print("log_id", log_id)
            # get the image file path
            for _, cam_name in enumerate(list(RingCameras)):
                if cam_name not in cameras_used:
                    print("skip ", cam_name, " camera")
                    continue
                cam_im_fpaths = loader.get_ordered_log_cam_fpaths(log_id, cam_name)

                # Sample every frame_sample_interval elements into a subset path list
                sampled_cam_im_fpaths = cam_im_fpaths[::frame_sample_interval]
                print("cam_im_fpaths = ", cam_im_fpaths)
                for i, im_fpath in enumerate(sampled_cam_im_fpaths):
                    try:
                        data_dict = {}
                        data_dict["image"] = str(im_fpath).replace(root_folder, "")
                        # get the object labels

                        cam_timestamp_ns = int(im_fpath.stem)
                        city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
                        if city_SE3_ego is None:
                            logger.exception("missing LiDAR pose")
                            continue

                        # load feather file path, e.g. '315978406032859416.feather"
                        lidar_fpath = loader.get_closest_lidar_fpath(
                            log_id, cam_timestamp_ns
                        )
                        if lidar_fpath is None:
                            logger.info(
                                "No LiDAR sweep found within the synchronization interval for %s, so skipping...",
                                cam_name,
                            )
                            continue

                        lidar_timestamp_ns = int(lidar_fpath.stem)

                        lidar_points_ego = io_utils.read_lidar_sweep(
                            lidar_fpath, attrib_spec="xyz"
                        )

                        (
                            uv,
                            points_cam,
                            is_valid_points,
                        ) = loader.project_ego_to_img_motion_compensated(
                            points_lidar_time=lidar_points_ego,
                            cam_name=cam_name,
                            cam_timestamp_ns=cam_timestamp_ns,
                            lidar_timestamp_ns=lidar_timestamp_ns,
                            log_id=log_id,
                        )

                        if is_valid_points is None or uv is None or points_cam is None:
                            continue

                        if is_valid_points.sum() == 0:
                            continue

                        uv_int: NDArrayInt = np.round(uv[is_valid_points]).astype(
                            np.int32
                        )  # image coordinates in pixels
                        points_cam = points_cam[
                            is_valid_points
                        ]  # 3d points in camera coordinates

                        # read the object bounding boxes and labels
                        cuboids = loader.get_labels_at_lidar_timestamp(
                            log_id, lidar_timestamp_ns
                        )

                        # convert to camera reference frame
                        # project cuboids to camera reference frame
                        pinhole_camera = loader.get_log_pinhole_camera(
                            log_id=log_id, cam_name=cam_name
                        )

                        city_SE3_ego_cam_t = loader.get_city_SE3_ego(
                            log_id=log_id, timestamp_ns=cam_timestamp_ns
                        )

                        # get transformation to bring point in egovehicle frame to city frame,
                        # at the time when the LiDAR sweep was recorded.
                        city_SE3_ego_lidar_t = loader.get_city_SE3_ego(
                            log_id=log_id, timestamp_ns=lidar_timestamp_ns
                        )

                        intrinsics = [
                            pinhole_camera.intrinsics.fx_px,
                            pinhole_camera.intrinsics.fy_px,
                            pinhole_camera.intrinsics.cx_px,
                            pinhole_camera.intrinsics.cy_px,
                            pinhole_camera.intrinsics.width_px,
                            pinhole_camera.intrinsics.height_px,
                        ]

                        # point clouds

                        # Ensure the number of points to sample does not exceed available points
                        num_points_to_sample = min(points_per_frame, len(uv_int))

                        # Calculate the interval for uniform sampling
                        sampled_indices = np.random.choice(
                            len(uv_int), num_points_to_sample, replace=False
                        )

                        # Subset the uv_int and points_cam arrays
                        uv_int = uv_int[sampled_indices].tolist()
                        points_cam = points_cam[sampled_indices].tolist()

                        data_dict["intrinsics"] = intrinsics
                        data_dict["pixel_coords"] = uv_int
                        # Read the image file path as a PIL image
                        undistorted_pil_image = Image.open(im_fpath)

                        # Check if the fx value in new_K is greater than 1000
                        if intrinsics[0] > 1000:
                            # Calculate the scaling factor to make fx equal to 1000
                            scale_factor = 1000 / intrinsics[0]

                            # Rescale the undistorted_pil_image
                            new_width = int(undistorted_pil_image.width * scale_factor)
                            new_height = int(
                                undistorted_pil_image.height * scale_factor
                            )
                            undistorted_pil_image = undistorted_pil_image.resize(
                                (new_width, new_height), Image.LANCZOS
                            )

                            # Rescale the pixel coordinates
                            data_dict["pixel_coords"] = [
                                (int(x * scale_factor), int(y * scale_factor))
                                for x, y in data_dict["pixel_coords"]
                            ]

                            data_dict["intrinsics"] = [
                                1000.0,
                                1000.0,
                                data_dict["intrinsics"][2] * scale_factor,
                                data_dict["intrinsics"][3] * scale_factor,
                                undistorted_pil_image.width,
                                undistorted_pil_image.height,
                            ]

                        # Construct the full path for the output image
                        output_image_path = os.path.join(
                            out_image_folder, data_dict["image"].lstrip("/")
                        )

                        # Create the directory if it doesn't exist
                        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                        # Save the undistorted image as a JPEG
                        undistorted_pil_image.save(output_image_path)

                        data_dict["depth"] = []

                        for point_id in range(len(uv_int)):
                            data_dict["depth"].append(
                                (
                                    points_cam[point_id][0] ** 2
                                    + points_cam[point_id][1] ** 2
                                    + points_cam[point_id][2] ** 2
                                )
                                ** 0.5
                            )

                        f.write(f"{json.dumps(data_dict)}\n")
                        count_rows += 1
                        # exit()
                        count += 1
                        if count % 1000 == 0:
                            print("data_dict", data_dict)
                            print(
                                "processed ", count, " frames and ", count_rows, "rows"
                            )
                    except Exception as e:
                        print("error ", e)
                        break
