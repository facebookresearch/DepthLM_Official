# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import imageio as iio
import lz4.block
import numpy as np
import yaml
from common.scene_release import ScannetppScene_Release
from common.utils.utils import load_json, load_yaml_munch, read_txt_list, run_command
from munch import Munch
from tqdm import tqdm


def extract_rgb(scene, w=512, h=384):
    scene.iphone_rgb_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {scene.iphone_video_path} -vf scale={w}:{h} -start_number 0 -q:v 1 {scene.iphone_rgb_dir}/frame_%06d.jpg"
    return run_command(cmd, verbose=True, exit_on_error=False)


def extract_masks(scene, w=512, h=384):
    scene.iphone_video_mask_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {str(scene.iphone_video_mask_path)} -pix_fmt gray -vf scale={w}:{h} -start_number 0 {scene.iphone_video_mask_dir}/frame_%06d.png"
    return run_command(cmd, verbose=True, exit_on_error=False)


def extract_depth(scene):
    # global compression with zlib
    height, width = 192, 256
    sample_rate = 1
    scene.iphone_depth_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(scene.iphone_depth_path, "rb") as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in tqdm(
            range(0, depth.shape[0], sample_rate), desc="decode_depth"
        ):
            iio.imwrite(
                f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png",
                (depth * 1000).astype(np.uint16),
            )
    # per frame compression with lz4/zlib
    except:
        frame_id = 0
        with open(scene.iphone_depth_path, "rb") as infile:
            while True:
                size = infile.read(4)  # 32-bit integer
                if len(size) == 0:
                    break
                size = int.from_bytes(size, byteorder="little")
                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                # read the whole file
                data = infile.read(size)
                try:
                    # try using lz4
                    data = lz4.block.decompress(
                        data, uncompressed_size=height * width * 2
                    )  # UInt16 = 2bytes
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                except:
                    # try using zlib
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                    depth = (depth * 1000).astype(np.uint16)

                # 6 digit frame id = 277 minute video at 60 fps
                iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png", depth)
                frame_id += 1


def main(args):
    cfg = load_yaml_munch(args.config_file)

    # get the scenes to process, specify any one
    if cfg.get("scene_list_file"):
        scene_ids = read_txt_list(cfg.scene_list_file)
    elif cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    elif cfg.get("splits"):
        scene_ids = []
    # Read only the immediate level subfolders of cfg.data_root as scene_ids
    scene_ids = [
        f
        for f in os.listdir(cfg.data_root + "data/")
        if os.path.isdir(os.path.join(cfg.data_root + "data/", f))
    ]

    print("Scene IDs:", scene_ids)
    print("Number of scenes:", len(scene_ids))

    output_dir = "/home/czptc2h/datasets/scannet_pp/out_images"
    output_dir_json = (
        "/home/czptc2h/datasets/scannet_pp/scannet_depth_instructions.jsonl"
    )

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    sample_interval = 10
    points_per_frame = 100
    image_width = 1280  # resize first to save memory
    image_height = 960

    # get the options to process
    # go through each scene
    # Open a new jsonl file at output_dir_json
    with open(output_dir_json, "w") as jsonl_file:
        for scene_id in tqdm(scene_ids, desc="scene"):
            try:
                scene = ScannetppScene_Release(
                    scene_id, data_root=Path(cfg.data_root) / "data"
                )

                print(
                    "cfg.data_root = ",
                    cfg.data_root,
                    "scene_id = ",
                    scene_id,
                    "scene = ",
                    scene,
                )

                # # extract data for the current scene
                out_rgb = extract_rgb(scene, image_width, image_height)
                if out_rgb.returncode != 0:
                    print("error during rgb extraction, go to the next scene")
                    continue
                out_mask = extract_masks(scene, image_width, image_height)
                if out_mask.returncode != 0:
                    print("error during mask extraction, go to the next scene")
                    continue
                extract_depth(scene)

                # convert data into json
                # remove all files in the folders
                # Iteratively read all png files under scene.iphone_video_mask_dir
                for i, (image_file, mask_file, depth_file) in enumerate(
                    zip(
                        os.listdir(scene.iphone_rgb_dir),
                        os.listdir(scene.iphone_video_mask_dir),
                        os.listdir(scene.iphone_depth_dir),
                    )
                ):
                    if i % sample_interval != 0:
                        continue
                    data_dict = {}

                    data_dict["image"] = (
                        str(scene.iphone_rgb_dir / image_file)
                        .replace(cfg.data_root, "")
                        .replace("data/", "")
                    )
                    # Move the image file to the specified output directory
                    destination_path = Path(output_dir) / data_dict["image"]
                    destination_path.parent.mkdir(
                        parents=True, exist_ok=True
                    )  # Ensure the directory exists
                    shutil.move(
                        str(scene.iphone_rgb_dir / image_file), destination_path
                    )

                    data_dict["intrinsics"] = [
                        1427.4375 * (image_width / 1920),
                        1427.4375 * (image_height / 1440),
                        959.5 * (image_width / 1920),
                        719.5 * (image_height / 1440),
                        image_width,
                        image_height,
                    ]  # rescaled intrinsics

                    if mask_file.endswith(".png"):
                        mask_path = scene.iphone_video_mask_dir / mask_file
                        mask_image = iio.imread(mask_path)
                        # Randomly sample points_per_frame pixels where mask_image is not 0
                        non_zero_indices = np.argwhere(mask_image != 0)
                        random_indices = np.random.choice(
                            non_zero_indices.shape[0],
                            points_per_frame,
                            replace=False,
                        )
                        sampled_indices = (
                            non_zero_indices
                            * np.array([192 / image_height, 256 / image_width])
                        ).astype(int)[random_indices]

                        sampled_indices_ori = (non_zero_indices).astype(int)[
                            random_indices
                        ]

                        data_dict["pixel_coords"] = [
                            [int(x), int(y)] for y, x in sampled_indices_ori
                        ]

                    # convert to euclidean distance
                    if depth_file.endswith(".png"):
                        depth_path = scene.iphone_depth_dir / depth_file
                        depth_image = iio.imread(depth_path)
                        fx, fy, cx, cy, _, _ = data_dict["intrinsics"]

                        fx, fy, cx, cy, _, _ = data_dict["intrinsics"]
                        pixel_coords = np.array(data_dict["pixel_coords"])
                        x = (pixel_coords[:, 0] - cx) / fx
                        y = (pixel_coords[:, 1] - cy) / fy
                        z = (
                            depth_image[sampled_indices[:, 0], sampled_indices[:, 1]]
                            / 1000.0
                        )

                        data_dict["depth"] = np.sqrt(x**2 + y**2 + z**2).tolist()
                        # Print samples to verify computation
                        sample_indices = np.random.choice(
                            len(z), min(5, len(z)), replace=False
                        )
                        for idx in sample_indices:
                            print(
                                f"Sample {idx}: z = {z[idx]}, pixel_coords = {data_dict['pixel_coords'][idx]}, depth = {data_dict['depth'][idx]}"
                            )


                    json.dump(data_dict, jsonl_file)
                    jsonl_file.write("\n")

                # Remove the folder scene.iphone_rgb_dir
                shutil.rmtree(scene.iphone_rgb_dir)
                shutil.rmtree(scene.iphone_video_mask_dir)
                shutil.rmtree(scene.iphone_depth_dir)

            except Exception as e:
                print(e)
                continue


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
