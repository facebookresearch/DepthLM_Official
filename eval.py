# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from utils.datasets import dataset_eval
from utils.metrics import *


def convert_example_pixtral(example, image_before_text=None):
    messages = []
    problem = example.get("problem")
    if "images" in example:
        images = example.get("images")

        if image_before_text is not None and image_before_text:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "content": problem}],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "content": problem}]
                    + [{"type": "image", "image": img} for img in images],
                }
            )
    else:
        image = example.get("image")
        if image_before_text is not None and image_before_text:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "content": problem},
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "content": problem},
                        {"type": "image", "image": image},
                    ],
                }
            )
    example["messages"] = messages
    return example


def convert_example(example, image_before_text=None):
    messages = []
    if "system" in example:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": example["system"]}],
            }
        )
    else:
        SYSTEM_PROMPT = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
            "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think><answer> answer here </answer>"
        )
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            }
        )
    problem = example.get("problem")
    if "images" in example:
        images = example.get("images")

        if image_before_text is not None and image_before_text:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "text": problem}],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": problem}]
                    + [{"type": "image", "image": img} for img in images],
                }
            )
    else:
        image = example.get("image")
        if image_before_text is not None and image_before_text:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": problem},
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": problem},
                        {"type": "image", "image": image},
                    ],
                }
            )
    example["messages"] = messages
    return example


def main(args):
    model_path = args.model_path
    img_fodler = args.image_folder
    json_path = args.json_path
    processor = AutoProcessor.from_pretrained(model_path)

    if "pixtral" in model_path.lower():
        print("loading DepthLM with pixtral (12B) architecture")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation={
                "text_config": "flash_attention_2",
                "vision_config": "eager",
            },
            device_map="auto",
        )
        model.eval()
    else:
        print("loading DepthLM with qwen2.5-vl architecture")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

    dataset = dataset_eval(
        json_path,
        img_fodler,
        normalized_focal_length=750.0,  # change to the corresponding value for other models
        randomize=not args.eval_no_random,
    )

    print(f"dataset size = {len(dataset)}")

    metric_funcs = [delta1_metric]

    metrics = []
    all_outputs = []  # List to store all answers
    all_solutions = []  # List to store all solutions

    samples_to_eval = (min(args.samples_to_eval, len(dataset)) // args.bsz) * args.bsz
    step = 1
    sampled_indices = list(range(0, len(dataset), step))[:samples_to_eval]

    with torch.no_grad():

        for i in tqdm(range(0, len(sampled_indices), args.bsz)):
            # for i in tqdm(range(args.bsz * 157, len(sampled_indices), args.bsz)):
            batch_indices = sampled_indices[i : i + args.bsz]

            batch_messages = [dataset[j] for j in batch_indices]

            if "pixtral" in model_path.lower():
                chat = [
                    convert_example_pixtral(msg, True)["messages"]
                    for msg in batch_messages
                ]

                inputs = processor.apply_chat_template(
                    chat,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                ).to("cuda", dtype=torch.bfloat16)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                batch_output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            else:
                # code for qwen based models
                if args.apply_system_prompt:
                    text = [
                        processor.apply_chat_template(
                            convert_example(msg, True)["messages"],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for msg in batch_messages
                    ]
                else:
                    batch_messages_text = [dataset[j]["prompt"] for j in batch_indices]
                    text = [
                        processor.apply_chat_template(
                            msg, tokenize=False, add_generation_prompt=True
                        )
                        for msg in batch_messages_text
                    ]

                    # image_inputs, video_inputs = process_vision_info(batch_messages)
                # breakpoint()
                image_inputs = [
                    x["images"] if "images" in x else x["image"] for x in batch_messages
                ]
                # print("image_inputs", image_inputs)
                if i == 0:
                    print(
                        "text = ",
                        text[0],
                        "apply_system_prompt = ",
                        args.apply_system_prompt,
                    )

                inputs = processor(
                    text=text,
                    images=image_inputs,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                # breakpoint()
                # Inference: Generation of the output
                # TODO maybe enable sampling here later
                generated_ids = model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    top_p=None,  # Unset top_p to avoid the warning
                    top_k=None,
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                batch_output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            print(f"model input = {batch_messages}")
            print(f"model output = {batch_output_text}")

            solution_list = [example["solution"] for example in batch_messages]
            for k, metric_func in enumerate(metric_funcs):
                if i == 0:
                    metrics.append(
                        metric_func(
                            batch_output_text,
                            solution_list.copy(),
                        )
                    )
                else:
                    metrics[k] += metric_func(
                        batch_output_text,
                        solution_list.copy(),
                    )

            all_outputs.extend(batch_output_text)
            all_solutions.extend(solution_list.copy())

    for i in range(len(metric_funcs)):
        print("final delta_1 = ", sum(metrics[i]) / len(metrics[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DepthLM parameters.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model."
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="./examples/ibims1/",
        help="folder that contains the image",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="./examples/ibims1/ibims1_val.jsonl",
        help="path to the meta data",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="maximum number of tokens to generate",
    )
    parser.add_argument("--bsz", type=int, default=1, help="Batch size for processing.")
    parser.add_argument(
        "--apply_system_prompt",
        action="store_true",
        help="For Qwen only, whether to apply system prompt or not.",
    )
    parser.add_argument(
        "--samples_to_eval",
        type=int,
        default=128,
        help="maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--eval_no_random",
        action="store_true",
        help="If true, turn off randomness in evaluation.",
    )
    args = parser.parse_args()

    main(args)
