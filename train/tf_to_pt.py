import argparse
import os
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
import torch
from audiodiffusion.audio_encoder import AudioEncoder
from keras.models import load_model
from torch import Tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt_model",
        type=str,
        default="models/audio-encoder",
        help="PyTorch model path",
    )
    parser.add_argument(
        "--tf_model",
        type=str,
        default="models/speccy_model",
        help="TensorFlow model path",
    )
    args = parser.parse_args()

    model = load_model(
        args.tf_model,
        custom_objects={"cosine_proximity": tf.compat.v1.keras.losses.cosine_proximity},
    )

    pytorch_model = AudioEncoder()
    new_state_dict = OrderedDict()
    for conv_block in range(3):
        new_state_dict[f"conv_blocks.{conv_block}.sep_conv.depthwise.weight"] = Tensor(
            model.get_layer(
                f"separable_conv2d_{conv_block + 1}"
            ).depthwise_kernel.numpy()
        ).permute(2, 3, 0, 1)
        new_state_dict[f"conv_blocks.{conv_block}.sep_conv.pointwise.weight"] = Tensor(
            model.get_layer(
                f"separable_conv2d_{conv_block + 1}"
            ).pointwise_kernel.numpy()
        ).permute(3, 2, 0, 1)
        new_state_dict[f"conv_blocks.{conv_block}.sep_conv.pointwise.bias"] = Tensor(
            model.get_layer(f"separable_conv2d_{conv_block + 1}").bias.numpy()
        )
        new_state_dict[f"conv_blocks.{conv_block}.batch_norm.weight"] = Tensor(
            model.get_layer(f"batch_normalization_{conv_block + 1}").gamma.numpy()
        )
        new_state_dict[f"conv_blocks.{conv_block}.batch_norm.running_mean"] = Tensor(
            model.get_layer(f"batch_normalization_{conv_block + 1}").moving_mean.numpy()
        )
        new_state_dict[f"conv_blocks.{conv_block}.batch_norm.running_var"] = Tensor(
            model.get_layer(
                f"batch_normalization_{conv_block + 1}"
            ).moving_variance.numpy()
        )
        new_state_dict[f"conv_blocks.{conv_block}.batch_norm.bias"] = Tensor(
            model.get_layer(f"batch_normalization_{conv_block + 1}").beta.numpy()
        )

    new_state_dict[f"dense_block.batch_norm.weight"] = Tensor(
        model.get_layer(f"batch_normalization_{conv_block + 2}").gamma.numpy()
    )
    new_state_dict[f"dense_block.batch_norm.running_mean"] = Tensor(
        model.get_layer(f"batch_normalization_{conv_block + 2}").moving_mean.numpy()
    )
    new_state_dict[f"dense_block.batch_norm.running_var"] = Tensor(
        model.get_layer(f"batch_normalization_{conv_block + 2}").moving_variance.numpy()
    )
    new_state_dict[f"dense_block.batch_norm.bias"] = Tensor(
        model.get_layer(f"batch_normalization_{conv_block + 2}").beta.numpy()
    )

    new_state_dict[f"dense_block.dense.weight"] = Tensor(
        model.get_layer(f"dense_1").kernel.numpy()
    ).permute(1, 0)
    new_state_dict[f"dense_block.dense.bias"] = Tensor(
        model.get_layer(f"dense_1").bias.numpy()
    )
    new_state_dict[f"embedding.weight"] = Tensor(
        model.get_layer(f"dense_2").kernel.numpy()
    ).permute(1, 0)
    new_state_dict[f"embedding.bias"] = Tensor(model.get_layer(f"dense_2").bias.numpy())

    pytorch_model.eval()
    pytorch_model.load_state_dict(new_state_dict, strict=False)
    pytorch_model.save_pretrained(args.pt_model)

    # test
    pytorch_model.eval()
    np.random.seed(42)
    example = np.random.random_sample((1, 96, 216, 1))
    with torch.no_grad():
        assert (
            np.abs(
                pytorch_model(Tensor(example).permute(0, 3, 1, 2)).numpy()
                - model(example).numpy()
            ).max()
            < 1e-3
        )
