import os
import glob
import argparse

import numpy as np
import torch

import qmf


def get_args():
    parser = argparse.ArgumentParser(
        description="Ablation study on QMF colorspace transform."
    )
    parser.add_argument(
        "--data", type=str, default="kodak", help="dataset name (default: kodak)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="?",
        help="path to dataset (default: data/{data})",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ablation_colorspace",
        help="save dir (default: ablation_colorspace)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ablation_colorspace",
        help="prefix for saved files (default: ablation_colorspace)",
    )

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"../data/{args.data}"

    return args


def eval_image(image):
    image_id = os.path.basename(image)
    image = qmf.read_image(image)

    results = []

    # QMF - RGB
    for quality in np.linspace(0.0, 10, 50):
        params = {
            "color_space": "RGB",
            "quality": quality,
            "patch": True,
            "patch_size": (8, 8),
            "bounds": (-16, 15),
            "dtype": torch.int8,
            "num_iters": 10,
            "verbose": False,
        }
        config = {"data": image_id, "method": "QMF", **params}
        log = qmf.eval_compression(image, qmf.qmf_encode, qmf.qmf_decode, **params)
        results.append({**config, **log})
        print(f"method QMF-RGB, image {image_id} done.")

    # QMF - YCbCr
    for quality in np.linspace(0, 40, 80):
        params = {
            "color_space": "YCbCr",
            "scale_factor": (0.5, 0.5),
            "quality": (quality, quality / 2, quality / 2),
            "patch": True,
            "patch_size": (8, 8),
            "bounds": (-16, 15),
            "dtype": torch.int8,
            "num_iters": 10,
            "verbose": False,
        }
        config = {"data": image_id, "method": "QMF", **params}
        log = qmf.eval_compression(image, qmf.qmf_encode, qmf.qmf_decode, **params)
        results.append({**config, **log})
        print(f"method QMF-YCbCr, image {image_id} done.")

    return results


def eval_dataset(data_dir):
    results = []
    for image in sorted(glob.glob(os.path.join(data_dir, "*.png"))):
        results.extend(eval_image(image))

    return results


if __name__ == "__main__":
    args = get_args()
    results = eval_dataset(args.data_dir)
    qmf.save_config(results, save_dir=args.save_dir, prefix=args.prefix)
