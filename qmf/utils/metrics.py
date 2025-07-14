from typing import Any
import functools
from operator import mul

import numpy as np
import torch
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
    multiscale_structural_similarity_index_measure,
)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Mean Absolute Error (MAE) between two tensors.

    Args:
        pred (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The MAE value.
    """
    return torch.mean(torch.abs(pred - target))


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Mean Squared Error (MSE) between two tensors.

    Args:
        pred (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        torch.Tensor: The MSE value.
    """
    return torch.mean((pred - target) ** 2)


def relative_error(
    pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-16
) -> torch.Tensor:
    """
    Calculate the relative error between two tensors.

    Args:
        pred (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.
        epsilon (float, optional): A small value to avoid division by zero (default: 1e-16).

    Returns:
        torch.Tensor: The relative error value.
    """
    numerator = torch.norm(pred - target, p=2)
    denominator = torch.norm(target, p=2)
    return numerator / (denominator + epsilon)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: int = 255) -> torch.Tensor:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        pred (torch.Tensor): The predicted image tensor.
        target (torch.Tensor): The target image tensor.
        data_range (int, optional): The data range of the images (default: 255).

    Returns:
        torch.Tensor: The PSNR value.
    """
    mse_value = mse(pred.float(), target.float())
    psnr_value = 20 * torch.log10(data_range / torch.sqrt(mse_value))
    return psnr_value


def ssim(
    pred: torch.Tensor, target: torch.Tensor, data_range: int = 255, **kwargs
) -> torch.Tensor:
    """
    Compute the Structural Similarity Index Measure (SSIM) between two images.

    Args:
        pred (torch.Tensor): The predicted image tensor of shape (C, H, W).
        target (torch.Tensor): The target image tensor of shape (C, H, W).
        data_range (int, optional): The data range of the images (default: 255).
        **kwargs: Additional keyword arguments for the SSIM metric.

    Returns:
        torch.Tensor: The SSIM index value.
    """
    return structural_similarity_index_measure(
        preds=pred.unsqueeze(0).float(),
        target=target.unsqueeze(0).float(),
        data_range=data_range,
        **kwargs
    )


def msssim(
    pred: torch.Tensor, target: torch.Tensor, data_range: int = 255, **kwargs
) -> torch.Tensor:
    """
    Compute the Multi-Scale Structural Similarity Index Measure (MS-SSIM) between two images.

    Args:
        pred (torch.Tensor): The predicted image tensor of shape (C, H, W).
        target (torch.Tensor): The target image tensor of shape (C, H, W).
        data_range (int, optional): The data range of the images (default: 255).
        **kwargs: Additional keyword arguments for the MS-SSIM metric.

    Returns:
        torch.Tensor: The MS-SSIM index value.
    """
    return multiscale_structural_similarity_index_measure(
        preds=pred.unsqueeze(0).float(),
        target=target.unsqueeze(0).float(),
        data_range=data_range,
        **kwargs
    )


def get_memory_usage(obj: Any) -> int:
    """
    Calculate the memory usage of an object containing NumPy arrays or PyTorch tensors in bytes.

    Args:
        obj (Any): The data structure.

    Returns:
        int: The memory usage of the data structure in bytes.
    """
    if isinstance(obj, (list, tuple, set)):
        return sum(get_memory_usage(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_memory_usage(item) for item in obj.values())
    elif isinstance(obj, bytes):
        return len(obj)
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    else:
        raise ValueError(
            "Unsupported data type. Please provide an object containing NumPy arrays or PyTorch tensors."
        )


def compression_ratio(input: Any, compressed: Any) -> float:
    """
    Calculate the compression ratio between the original and compressed data.

    Args:
        input (Any): The original data structure.
        compressed (Any): The compressed data structure.

    Returns:
        float: The compression ratio.
    """
    input_memory = get_memory_usage(input)
    compressed_memory = get_memory_usage(compressed)
    return input_memory / compressed_memory


def prod(x: list[int]) -> int:
    """
    Calculate the product of elements in a list.

    Args:
        x (list[int]): A list of integers.

    Returns:
        int: The product of the list elements.
    """
    return functools.reduce(mul, x, 1)


def bits_per_pixel(size: tuple[int, int, int], compressed: object) -> float:
    """
    Calculate the bits per pixel for a compressed image.

    Args:
        size (tuple[int, int, int]): The size of the original image (C, H, W).
        compressed (object): The compressed data structure.

    Returns:
        float: The bits per pixel value.
    """
    num_pixels = prod(size)
    compressed_memory = get_memory_usage(compressed)
    return compressed_memory * 8 / num_pixels
