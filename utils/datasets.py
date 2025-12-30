# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import json
import os
import random

import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


# unified prompt that can be used for both SFT and GRPO, our method is not sensitive to the prompt, so you can adjust it flexibly
def generate_prompt_depth_sft(
    depth,
    is_eval=False,
):
    problem = "Given this image, how far is the point pointed by the red arrow from the camera? Output the thinking process in <think> </think> and final answer (the meter number only, without the unit) in <answer> </answer> tags."

    thinking = (
        f"<think> The point is around {depth:.2f} meters away from the camera. </think>"
    )
    if is_eval:
        solution = f"<answer> {depth} </answer>"
    else:
        solution = f"<answer> {depth:.2f} </answer>"
    return problem, thinking, solution


# ####################### handle camera ambiguities ################################
def undistort_image(intrinsics: list, image: Image):
    # Check if fx and fy are not the same
    if abs(intrinsics[0] - intrinsics[1]) > 1e-3:
        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Create camera matrix from intrinsics
        camera_matrix = np.array(
            [
                [intrinsics[0], 0, intrinsics[2]],
                [0, intrinsics[1], intrinsics[3]],
                [0, 0, 1],
            ]
        )

        # Assume no distortion coefficients
        dist_coeffs = np.zeros((4, 1))

        # Get optimal new camera matrix
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (image_np.shape[1], image_np.shape[0]),
            1,
            (image_np.shape[1], image_np.shape[0]),
        )

        # Undistort the image
        undistorted_image_np = cv2.undistort(
            image_np, camera_matrix, dist_coeffs, None, new_camera_matrix
        )

        # Extract [fx, fy, cx, cy] from the new camera matrix
        new_intrinsics = [
            float(new_camera_matrix[0, 0]),
            float(new_camera_matrix[1, 1]),
            float(new_camera_matrix[0, 2]),
            float(new_camera_matrix[1, 2]),
        ]

        # Convert back to PIL image
        return Image.fromarray(undistorted_image_np), new_intrinsics
    else:
        return image, intrinsics


def normalizing_focal_length(
    normalized_focal_length: float, intrinsics: list, image: Image
):
    # Calculate the scaling factor for the focal length normalization
    scale_factor = normalized_focal_length / intrinsics[0]
    # Resize the image according to the scaling factor
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    # Update the intrinsics with the normalized focal length
    intrinsics = [
        intrinsics[0] * scale_factor,
        intrinsics[1] * scale_factor,
        intrinsics[2] * scale_factor,
        intrinsics[3] * scale_factor,
        new_width,
        new_height,
    ]

    return image.resize((new_width, new_height)), intrinsics


def is_within_range(coord, crop_range):
    x, y = coord
    left, top, right, bottom = crop_range
    return left <= x < right and top <= y < bottom


def adjust_index(
    index,
    pixel_coords,
):
    # Check if the current index is valid
    if pixel_coords[index] != [-1, -1]:
        return index

    # Search for the closest valid index
    left = index - 1
    right = index + 1
    n = len(pixel_coords)

    while left >= 0 or right < n:
        if left >= 0 and pixel_coords[left] != [-1, -1]:
            return left
        if right < n and pixel_coords[right] != [-1, -1]:
            return right
        left -= 1
        right += 1

    # If no valid index is found, return -1
    return -1


class dataset_eval(Dataset):
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        points_per_image=None,
        normalized_focal_length=1000.0,  # set to the intrinsics after original resize if needed
        randomize=True,  # randomly use pixels from each image
    ) -> None:
        super(dataset_eval, self).__init__()
        self.normalized_focal_length = normalized_focal_length

        print("reading data from ", data_path, "image_folder = ", image_folder)
        if ".jsonl" in data_path:
            import pandas as pd

            self.list_data_dict = pd.read_json(data_path, lines=True).to_dict(
                orient="records"
            )
        else:
            self.list_data_dict = json.load(open(data_path, "r"))

        self.data_path = data_path
        self.image_folder = image_folder
        self.length = self._get_length()

        if "scannet" in data_path:
            self.list_data_dict = self.list_data_dict[
                int(len(self.list_data_dict) * 0.98) :
            ]  # keep the last 2% for evaluation

            random.seed(42)
            random.shuffle(self.list_data_dict)

        self.random_indices = []

        if randomize:
            random.seed(42)  # Set a fixed seed for replicability
            while len(self.random_indices) < self.__len__():
                i = random.sample(range(len(self.list_data_dict[0]["pixel_coords"])), 1)
                self.random_indices.append((i))
        else:
            while len(self.random_indices) < self.__len__():
                for i in range(len(self.list_data_dict[0]["pixel_coords"])):
                    self.random_indices.append(([i]))
                    if len(self.random_indices) >= self.__len__():
                        break

    def _get_length(self) -> int:
        return len(self.list_data_dict)

    def __len__(self) -> int:
        return len(self.list_data_dict) * len(self.list_data_dict[0]["pixel_coords"])

    def extract_image_and_meta(self, index):
        index_ori = index

        index //= len(self.list_data_dict[0]["pixel_coords"])  # Image index
        random_index = self.random_indices[index_ori][0]  # Pixel index inside image

        # read image
        data_dict = {}
        data_dict["image"] = Image.open(
            os.path.join(
                self.image_folder, self.list_data_dict[index]["image"].lstrip("/")
            )
        )

        intrinsics = self.list_data_dict[index]["intrinsics"][:4]
        if intrinsics[0] == 0.0:  # handle intrinsic errors
            intrinsics[0] = intrinsics[1]
        if intrinsics[1] == 0.0:
            intrinsics[1] = intrinsics[0]

        data_dict["image"], intrinsics_new = undistort_image(
            intrinsics, data_dict["image"]
        )

        data_dict["image"], intrinsics_new = normalizing_focal_length(
            self.normalized_focal_length, intrinsics_new, data_dict["image"]
        )

        pixel_coords = [
            [
                int(
                    (coord[0] - intrinsics[2]) * (intrinsics_new[0] / intrinsics[0])
                    + intrinsics_new[2]
                ),
                int(
                    (coord[1] - intrinsics[3]) * (intrinsics_new[1] / intrinsics[1])
                    + intrinsics_new[3]
                ),
            ]
            for coord in self.list_data_dict[index]["pixel_coords"]
        ]

        pixel_coord = pixel_coords[
            random_index
        ].copy()  # pixel coords starts from top-left corner
        depth = self.list_data_dict[index]["depth"][random_index]

        # randomly decide the task and run the prompt generation functions
        # Adjustable cross size
        cross_size = 5  # You can modify this value to change the cross size
        cross_thickness = 1  # You can modify this value to change the cross thickness

        # Calculate the scaling factor
        scale_x = 1
        scale_y = 1

        # Scale the pixel coordinates
        scaled_pixel_x = int(pixel_coord[0] * scale_x)
        scaled_pixel_y = int(pixel_coord[1] * scale_y)
        center_x = round(intrinsics_new[2])
        center_y = round(intrinsics_new[3])

        # Compute the number of pixels from the center to scaled_pixel_x and scaled_pixel_y
        pixels_from_center_x = abs(scaled_pixel_x - center_x)
        pixels_from_center_y = abs(scaled_pixel_y - center_y)

        # Check if the adjustable cross can be drawn
        if (
            cross_size <= scaled_pixel_x < data_dict["image"].width - cross_size
            and cross_size <= scaled_pixel_y < data_dict["image"].height - cross_size
        ):
            # Draw a --> like arrow
            for dx in range(1, cross_size + 1):
                data_dict["image"].putpixel(
                    (scaled_pixel_x - dx, scaled_pixel_y), (255, 0, 0)
                )  # Horizontal line
            # Draw the arrowhead
            for dy in range(1, cross_size // 2 + 1):
                data_dict["image"].putpixel(
                    (
                        scaled_pixel_x - dy - 1,
                        scaled_pixel_y + dy,
                    ),
                    (255, 0, 0),
                )
                data_dict["image"].putpixel(
                    (
                        scaled_pixel_x - dy - 1,
                        scaled_pixel_y - dy,
                    ),
                    (255, 0, 0),
                )
        else:
            # Skip this sample and get the next one
            return self.extract_image_and_meta((index_ori + 1) % self.__len__())

        return (
            data_dict["image"],
            depth,
            pixel_coord,
            intrinsics_new,
        )

    def __getitem__(self, index):
        data_dict = {}
        (
            data_dict["image"],
            depth,
            pixel_coord,
            intrinsics,
        ) = self.extract_image_and_meta(index)

        # generate prompt
        data_dict["problem"], data_dict["thinking"], data_dict["solution"] = (
            generate_prompt_depth_sft(
                depth,
                is_eval=True,
            )
        )

        data_dict["pixel_coord"] = pixel_coord
        data_dict["intrinsics"] = intrinsics

        data_dict["system"] = "You are a helpful assistant."

        data_dict["prompt"] = [
            {
                "content": [
                    {"image": data_dict["image"], "type": "image"},
                    {"text": data_dict["problem"], "type": "text"},
                ],
                "role": "user",
            }
        ]
        return data_dict


class dataset_train(Dataset):
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        height_max=1200,
        height_min=700,
        width_max=1400,
        width_min=1000,
        normalized_focal_length=1000,
        sample_weights=None,  # support weighted sampling
        ratio_min=1.0,  # taskonomy dataset has intrinsic noise, we randomly rescale the aspect ratio of the images to handle that
        ratio_max=1.3,
    ) -> None:
        super().__init__()
        print("reading data from ", data_path, "image_folder = ", image_folder)
        data_paths = data_path.split(";")
        image_folders = image_folder.split(";")

        self.list_data_dict = []

        for dp in data_paths:
            if ".jsonl" in dp:
                import pandas as pd

                print("reading jsonl from ", dp)
                try:
                    self.list_data_dict.append(
                        pd.read_json(dp, lines=True).to_dict(orient="records")
                    )
                except Exception as e:
                    print(e)
                    self.list_data_dict.append(json.load(open(dp, "r")))
            else:
                self.list_data_dict.append(json.load(open(dp, "r")))

            if "scannet" in dp:
                self.list_data_dict[-1] = self.list_data_dict[-1][
                    : int(len(self.list_data_dict[-1]) * 0.98)
                ]

        self.data_path = data_paths
        self.image_folder = image_folders
        self.length = self._get_length()
        print(
            "reading finished, dataset size is ",
            self.__len__(),
            ", data_path = ",
            self.data_path,
            ", image_folder = ",
            self.image_folder,
        )
        self.random_indices = []
        self.normalized_focal_length = normalized_focal_length
        self.width_range = [width_min, width_max]
        self.height_range = [height_min, height_max]
        self.sample_weights = (
            [int(x) for x in sample_weights.split(";")]
            if sample_weights
            else [1] * len(self.list_data_dict)
        )
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max

    def _get_length(self) -> int:
        length = 0
        for data_dict in self.list_data_dict:
            length += len(data_dict)
        return length

    def __len__(self, ori_length=False) -> int:
        if ori_length:
            length = 0
            for data_dict in self.list_data_dict:
                length += len(data_dict)
            return length
        else:
            length = 0
            for data_dict in self.list_data_dict:
                length += (
                    len(data_dict) * 100
                )  # 100 labeled points per image in our data curation pipeline, cna change this number accordingly
            return length

    def getitem_Taskonomy(
        self, index, id_dataset
    ):  # taskonomy dataset has intrinsic noise, we randomly rescale the aspect ratio of the images to handle that
        index = index % len(self.list_data_dict[id_dataset])

        # read image
        data_dict = {}

        data_dict["image"] = Image.open(
            os.path.join(
                self.image_folder[id_dataset],
                self.list_data_dict[id_dataset][index]["image"].lstrip("/"),
            )
        )
        intrinsics = self.list_data_dict[id_dataset][index]["intrinsics"][:4]

        data_dict["image"], intrinsics_new = undistort_image(
            intrinsics, data_dict["image"]
        )

        if self.normalized_focal_length > 0:
            data_dict["image"], intrinsics_new = normalizing_focal_length(
                self.normalized_focal_length, intrinsics_new, data_dict["image"]
            )
        # Calculate the new height to maintain the aspect ratio of 1.3
        new_height = int(
            data_dict["image"].width / random.uniform(self.ratio_min, self.ratio_max)
        )

        # Resize the image
        data_dict["image"] = data_dict["image"].resize(
            (data_dict["image"].width, new_height)
        )

        # Adjust the intrinsics to account for the new image height
        intrinsics_new[1] *= new_height / intrinsics_new[5]  # Scale fy
        intrinsics_new[3] *= new_height / intrinsics_new[5]  # Scale cy
        intrinsics_new[5] = new_height  # Update height

        pixel_coords = [
            [
                int(
                    (coord[0] - intrinsics[2]) * (intrinsics_new[0] / intrinsics[0])
                    + intrinsics_new[2]
                ),
                int(
                    (coord[1] - intrinsics[3]) * (intrinsics_new[1] / intrinsics[1])
                    + intrinsics_new[3]
                ),
            ]
            for coord in self.list_data_dict[id_dataset][index]["pixel_coords"]
        ]

        if len(self.list_data_dict[id_dataset][index]["pixel_coords"]) - 1 > 0:
            random_index = random.randint(
                0, len(self.list_data_dict[id_dataset][index]["pixel_coords"]) - 1
            )
        else:
            print("no pixel in ", index, ": ", self.list_data_dict[id_dataset][index])
            return self.__getitem__((index + 1) % self.__len__())

        pixel_coord = pixel_coords[random_index]
        depth = self.list_data_dict[id_dataset][index]["depth"][random_index]

        # Adjustable cross size
        cross_size = 5  # You can modify this value to change the cross size
        # Calculate the scaling factor
        scale_x = 1
        scale_y = 1

        # Scale the pixel coordinates
        scaled_pixel_x = int(pixel_coord[0] * scale_x)
        scaled_pixel_y = int(pixel_coord[1] * scale_y)

        # Check if the adjustable cross can be drawn
        if (
            cross_size <= scaled_pixel_x < data_dict["image"].width - cross_size
            and cross_size <= scaled_pixel_y < data_dict["image"].height - cross_size
        ):
            # Draw a --> like arrow
            for dx in range(1, cross_size + 1):
                data_dict["image"].putpixel(
                    (scaled_pixel_x - dx, scaled_pixel_y), (255, 0, 0)
                )  # Horizontal line
            # Draw the arrowhead
            for dy in range(1, cross_size // 2 + 1):
                data_dict["image"].putpixel(
                    (
                        scaled_pixel_x - dy - 1,
                        scaled_pixel_y + dy,
                    ),
                    (255, 0, 0),
                )
                data_dict["image"].putpixel(
                    (
                        scaled_pixel_x - dy - 1,
                        scaled_pixel_y - dy,
                    ),
                    (255, 0, 0),
                )
        else:
            # Skip this sample and get the next one
            return self.__getitem__((index + 1) % self.__len__())

        # generate prompt
        data_dict["problem"], data_dict["thinking"], data_dict["solution"] = (
            generate_prompt_depth_sft(depth)
        )

        data_dict["system"] = "You are a helpful assistant."
        return data_dict

    def getitem_noTaskonomy(self, index, id_dataset):
        index_ori = index
        index = index % len(self.list_data_dict[id_dataset])
        intrinsics = self.list_data_dict[id_dataset][index]["intrinsics"][:4]

        # read image
        data_dict = {}

        img = Image.open(
            os.path.join(
                self.image_folder[id_dataset],
                self.list_data_dict[id_dataset][index]["image"].lstrip("/"),
            )
        )

        img, intrinsics_new = undistort_image(intrinsics, img)

        if self.normalized_focal_length > 0:
            img, intrinsics_new = normalizing_focal_length(
                self.normalized_focal_length, intrinsics_new, img
            )

        data_dict["image"] = img

        pixel_coords = [
            [
                int(
                    (coord[0] - intrinsics[2]) * (intrinsics_new[0] / intrinsics[0])
                    + intrinsics_new[2]
                ),
                int(
                    (coord[1] - intrinsics[3]) * (intrinsics_new[1] / intrinsics[1])
                    + intrinsics_new[3]
                ),
            ]
            for coord in self.list_data_dict[id_dataset][index]["pixel_coords"]
        ]

        # Random center crop
        width, height = data_dict["image"].size

        crop_height = int(
            min(height, random.uniform(self.height_range[0], self.height_range[1]))
        )
        crop_width = int(
            min(width, random.uniform(self.width_range[0], self.width_range[1]))
        )

        center_x = round(intrinsics_new[2])
        center_y = round(intrinsics_new[3])

        # Ensure the crop is within the specified bounds
        left = max(0, (width - crop_width) // 2)
        top = max(
            0,
            (height - crop_height) // 2,
        )
        right = min(width, left + crop_width)
        bottom = min(height, top + crop_height)

        data_dict["image"] = data_dict["image"].crop((left, top, right, bottom))

        # Adjust intrinsics_new to account for cropping
        intrinsics_new[2] -= left  # Adjust cx
        intrinsics_new[3] -= top  # Adjust cy
        intrinsics_new[4] = data_dict["image"].width  # Update width
        intrinsics_new[5] = data_dict["image"].height  # Update height

        pixel_coords = [
            (
                [coord[0] - left, coord[1] - top]
                if is_within_range(coord, (left, top, right, bottom))
                else [-1, -1]
            )
            for coord in pixel_coords
        ]

        if len(self.list_data_dict[id_dataset][index]["pixel_coords"]) - 1 > 0:
            random_index = random.randint(
                0, len(self.list_data_dict[id_dataset][index]["pixel_coords"]) - 1
            )
        else:
            print("no pixel in ", index, ": ", self.list_data_dict[id_dataset][index])
            return self.__getitem__((index + 1) % self.__len__(True))

        random_index = adjust_index(random_index, pixel_coords)
        if random_index == -1:
            # Skip this sample and get the next one
            return self.__getitem__((index_ori + 1) % self.__len__(True))

        pixel_coord = pixel_coords[random_index].copy()
        depth = self.list_data_dict[id_dataset][index]["depth"][random_index]

        # Adjustable cross size
        cross_size = 5  # You can modify this value to change the cross size

        # Scale the pixel coordinates
        scaled_pixel_x = int(pixel_coord[0])
        scaled_pixel_y = int(pixel_coord[1])

        # Check if the adjustable cross can be drawn
        if (
            cross_size <= scaled_pixel_x < data_dict["image"].width - cross_size
            and cross_size <= scaled_pixel_y < data_dict["image"].height - cross_size
        ):
            # Draw a --> like arrow
            for dx in range(1, cross_size + 1):
                data_dict["image"].putpixel(
                    (scaled_pixel_x - dx, scaled_pixel_y), (255, 0, 0)
                )  # Horizontal line
            # Draw the arrowhead
            for dy in range(1, cross_size // 2 + 1):
                data_dict["image"].putpixel(
                    (
                        scaled_pixel_x - dy - 1,
                        scaled_pixel_y + dy,
                    ),
                    (255, 0, 0),
                )
                data_dict["image"].putpixel(
                    (
                        scaled_pixel_x - dy - 1,
                        scaled_pixel_y - dy,
                    ),
                    (255, 0, 0),
                )
        else:
            return self.__getitem__((index + 1) % self.__len__(True))

        data_dict["problem"], data_dict["thinking"], data_dict["solution"] = (
            generate_prompt_depth_sft(depth)
        )

        data_dict["system"] = "You are a helpful assistant."
        return data_dict

    def __getitem__(self, index):
        id_dataset = random.choices(
            range(len(self.list_data_dict)), weights=self.sample_weights, k=1
        )[0]
        if "taskonomy" in self.image_folder[id_dataset]:
            return self.getitem_Taskonomy(index, id_dataset)
        else:
            return self.getitem_noTaskonomy(index, id_dataset)
