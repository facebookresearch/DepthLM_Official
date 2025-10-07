# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
import trl

from qwen_vl_utils import process_vision_info


# from torch import distributed as dist

from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    LlavaForConditionalGeneration,
    MllamaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    set_seed,
    TrainerCallback,
)
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
)

from utils.datasets import dataset_train
from utils.callbacks import get_callbacks
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig(trl.ModelConfig):
    output_model_local_path: str = field(
        default="test-output",
        metadata={"help": "Output model local path, do not set manually"},
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to push to a Hub revision/branch."}
    )


@dataclass
# pyre-fixme[11]: Annotation `ScriptArguments` is not defined as a type.
class SFTScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    dataset_class: str = field(
        default="LazySupervisedDataset_ArgoverseDepth_GRPO",
        metadata={"help": "dataset class name in callm.reason.openr1.utils.datasets"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "image folder on manifold"},
    )
    augment: Optional[float] = field(
        default=None,
        metadata={"help": "augmentation ratio"},
    )
    normalized_focal_length: Optional[float] = field(
        default=None,
        metadata={"help": "normalized focal length"},
    )
    sample_weights: Optional[str] = field(
        default=None,
        metadata={"help": "weights for sampling"},
    )
    pad: Optional[bool] = field(
        default=None,
        metadata={
            "help": "whether to pad image to have same width and height in 2 image strategy"
        },
    )
    height_max: Optional[float] = field(
        default=None,
        metadata={"help": "max height"},
    )
    height_min: Optional[float] = field(
        default=None,
        metadata={"help": "min height"},
    )
    width_min: Optional[float] = field(
        default=None,
        metadata={"help": "min width"},
    )
    width_max: Optional[float] = field(
        default=None,
        metadata={"help": "max width"},
    )
    ratio_min: Optional[float] = field(
        default=None,
        metadata={"help": "min ratio"},
    )
    ratio_max: Optional[float] = field(
        default=None,
        metadata={"help": "max ratio"},
    )


processor = None


def configure_pixtral_vision_tower(model, compute_dtype, device):
    vision_tower = model.vision_tower
    vision_tower.to(dtype=compute_dtype, device=device)


def convert_example(example):
    """
    correct example into "messages"
    eg:
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<think>\nI can see 10 objects\n</think>\n<answer>\n10\n</answer>"}
      ]
    }
    """
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

    thinking = example.get("thinking", "")  # no thinking case included
    problem = example.get("problem")
    solution = example.get("solution")
    if "images" in example:
        images = example.get("images")
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": problem}],
            }
        )
    else:
        image = example.get("image")
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": problem},
                ],
            }
        )
    messages.append(
        {
            "role": "assistant",
            "content": f"{thinking}\n\n{solution}",
        }
    )

    example["messages"] = messages
    return example


def convert_example_phi4(example):
    """
    correct example into "messages"
    eg:
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<think>\nI can see 10 objects\n</think>\n<answer>\n10\n</answer>"}
      ]
    }
    """
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

    thinking = example.get("thinking", "")  # no thinking case included
    problem = example.get("problem")
    solution = example.get("solution")
    if "images" in example:
        images = example.get("images")
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": problem}],
            }
        )
    else:
        image = example.get("image")
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": problem},
                ],
            }
        )
    messages.append(
        {
            "role": "assistant",
            "content": f"{thinking}\n\n{solution}",
        }
    )

    example["messages"] = messages
    return example


def pad_sequence(sequences, padding_side="right", padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), "All tensors must have the same number of dimensions"

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def pmc_vqa_collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []
    for inputs in batch:
        input_ids_list.append(inputs["input_ids"][0])
        labels_list.append(inputs["labels"][0])
        input_image_embeds_list.append(inputs["input_image_embeds"])
        image_attention_mask_list.append(inputs["image_attention_mask"])
        image_sizes_list.append(inputs["image_sizes"])

    input_ids = pad_sequence(input_ids_list, padding_side="right", padding_value=0)
    labels = pad_sequence(labels_list, padding_side="right", padding_value=0)
    attention_mask = (input_ids != 0).long()
    input_image_embeds = cat_with_pad(input_image_embeds_list, dim=0)
    image_attention_mask = cat_with_pad(image_attention_mask_list, dim=0)
    image_sizes = torch.cat(image_sizes_list)

    # breakpoint()
    return BatchFeature(
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_image_embeds": input_image_embeds,
            "image_attention_mask": image_attention_mask,
            "image_sizes": image_sizes,
            "input_mode": 1,  # vision mode
        }
    )


def collate_fn_phi4(examples):
    _IGNORE_INDEX = -100
    _MAX_TRAINING_LENGTH = 8192
    batch = []
    for example in examples:
        image = example["image"]
        question = example.get("problem")
        user_message = {
            "role": "user",
            "content": "<|image_1|>" + question,
        }
        prompt = processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{example.get("thinking", "")}\n\n{example.get("solution")}<|end|><|endoftext|>'
        inputs = processor(prompt, images=[image], return_tensors="pt")

        answer_ids = processor.tokenizer(answer, return_tensors="pt").input_ids

        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1] :] = answer_ids

        # breakpoint()
        if input_ids.size(1) > _MAX_TRAINING_LENGTH:
            input_ids = input_ids[:, :_MAX_TRAINING_LENGTH]
            labels = labels[:, :_MAX_TRAINING_LENGTH]
            if torch.all(labels == _IGNORE_INDEX).item():
                # workaround to make sure loss compute won't fail
                labels[:, -1] = processor.tokenizer.eos_token_id
        batch.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "input_image_embeds": inputs.input_image_embeds,
                "image_attention_mask": inputs.image_attention_mask,
                "image_sizes": inputs.image_sizes,
            }
        )

    return pmc_vqa_collate_fn(batch)


def find_subsequence(sequence, subsequence):
    """
    Helper function to find the starting index of a subsequence within a sequence.
    """
    seq_len = len(sequence)
    sub_len = len(subsequence)
    for i in range(seq_len - sub_len + 1):
        if torch.equal(sequence[i : i + sub_len], subsequence):
            return i
    return None


def get_image_token_count(image, dummy_text="describe this image"):
    """
    Compute the number of tokens generated for an image using the model's vision tower.
    Returns 0 if token computation fails.
    """
    try:
        inputs = processor(images=image, text=dummy_text, return_tensors="pt").to(
            "cuda"
        )
        with torch.no_grad():
            output = model.vision_tower(pixel_values=inputs["pixel_values"])
        token_count = output.last_hidden_state.shape[1]
        if token_count == 0:
            raise ValueError("Image token count is zero.")
        return token_count
    except Exception as e:
        print(f"[ERROR] Failed to compute image tokens: {e}")
        return 0  # Return zero to flag as invalid


def collate_fn_pixtral(examples):
    texts = [
        processor.apply_chat_template(
            convert_example(example)["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in examples
    ]
    image_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    # print("texts = ", texts[0])
    # breakpoint()
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return batch


def collate_fn(examples):
    # breakpoint()
    texts = [
        processor.apply_chat_template(
            convert_example(example)["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in examples
    ]
    image_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    # print("texts = ", texts[0])
    # breakpoint()
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels
    # breakpoint()
    return batch


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")


    print("script_args.image_folder = ", script_args.image_folder)
    training_args.output_dir = model_args.output_model_local_path

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    dataset_kwargs = {}

    if script_args.normalized_focal_length is not None:
        dataset_kwargs["normalized_focal_length"] = script_args.normalized_focal_length
    if script_args.sample_weights is not None:
        dataset_kwargs["sample_weights"] = ";".join(
            weight
            for i, weight in enumerate(script_args.sample_weights.split(";"))
        )
    if script_args.height_max is not None:
        dataset_kwargs["height_max"] = script_args.height_max
    if script_args.height_min is not None:
        dataset_kwargs["height_min"] = script_args.height_min
    if script_args.width_min is not None:
        dataset_kwargs["width_min"] = script_args.width_min
    if script_args.width_max is not None:
        dataset_kwargs["width_max"] = script_args.width_max
    if script_args.ratio_min is not None:
        dataset_kwargs["ratio_min"] = script_args.ratio_min
    if script_args.ratio_max is not None:
        dataset_kwargs["ratio_max"] = script_args.ratio_max

    dataset = dataset_train(script_args.dataset_name, script_args.image_folder, **dataset_kwargs)


    print("[dataset] dataset_size = ", len(dataset))

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )
        logger.info("Using AutoProcessor for vision-language model.")
        if hasattr(processor, "pad_token") and processor.pad_token is None:
            processor.pad_token = processor.eos_token
        elif (
            hasattr(processor.tokenizer, "pad_token")
            and processor.tokenizer.pad_token is None
        ):
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    elif "pixtral-12b" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        )

        if hasattr(processor, "pad_token") and processor.pad_token is None:
            processor.pad_token = processor.eos_token
        elif (
            hasattr(processor.tokenizer, "pad_token")
            and processor.tokenizer.pad_token is None
        ):
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        processor.image_processor.do_resize = False
        processor.image_processor.do_rescale = False
        # breakpoint()

    else:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
        logger.info("Using AutoProcessor.")

    # ###################
    # # Model init kwargs
    # ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    if "pixtral-12b" in model_args.model_name_or_path.lower():
        # seems like use_cache is not supported in the model class
        model_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation={
                "text_config": "flash_attention_2",
                "vision_config": "eager",
            },
            torch_dtype=torch_dtype,
            device_map=(
                get_kbit_device_map() if quantization_config is not None else None
            ),
            quantization_config=quantization_config,
        )
    else:
        # training_args.model_init_kwargs = model_kwargs
        model_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=(
                get_kbit_device_map() if quantization_config is not None else None
            ),
            quantization_config=quantization_config,
        )

    if "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "pixtral-12b" in model_args.model_name_or_path.lower():
        model = LlavaForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
            # This is a workaround for a bug in the current implementation of gradient checkpointing
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )

    ############################
    # Initialize the SFT Trainer
    ############################

    callbacks = get_callbacks(training_args, model_args)
    # # configure TensorboardCallback to upload to manifold
    callbacks.append(
        TensorBoardCallback(
            SummaryWriter(
                log_dir=os.path.join(
                    training_args.output_dir,
                    "tensorboard_logs",
                ),
                comment="",
                purge_step=None,
                max_queue=10,
                flush_secs=120,
                filename_suffix=str(uuid.uuid4()),
            )
        )
    )

    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False

    if "pixtral" in model_args.model_name_or_path.lower():
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=processor.tokenizer,
            data_collator=collate_fn_pixtral,
            peft_config=get_peft_config(model_args),
            callbacks=callbacks,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=processor.tokenizer,
            data_collator=collate_fn,
            peft_config=get_peft_config(model_args),
            callbacks=callbacks,
        )

    # ###############
    # # Training loop
    # ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ##################################
    # # Save model and create model card
    # ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    output_model_basename = os.path.basename(model_args.output_model_filename)
    model_args.output_model_local_path = os.path.join(
        training_args.output_dir,
        "models",
        "DepthLM",
    )
    os.makedirs(model_args.output_model_local_path, exist_ok=True)

    main(script_args, training_args, model_args)
