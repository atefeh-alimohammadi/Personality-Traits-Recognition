#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import numpy as np
import torch

# Function to create one-hot encoded tensors
def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    """
    Convert target indices to one-hot encoded format.

    Args:
        x (Tensor): Indices tensor.
        num_classes (int): Number of classes.
        on_value (float): Value for active class.
        off_value (float): Value for inactive classes.
        device (str): Device to create tensor on.

    Returns:
        Tensor: One-hot encoded tensor.
    """
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

# Function to mixup targets
def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    """
    Create mixup labels with optional label smoothing.

    Args:
        target (Tensor): Target tensor.
        num_classes (int): Number of classes.
        lam (float): Mixup lambda value.
        smoothing (float): Label smoothing factor.
        device (str): Device to create tensor on.

    Returns:
        Tensor: Mixed target tensor.
    """
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)


# In[2]:


# Function to calculate random bounding box for CutMix
def rand_bbox(img_shape, lam, margin=0., count=None):
    """
    Generate random bounding box for CutMix.

    Args:
        img_shape (tuple): Shape of the image.
        lam (float): Lambda value for CutMix.
        margin (float): Margin to limit bbox edges.
        count (int): Number of bounding boxes to generate.

    Returns:
        Tuple: Bounding box coordinates.
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh

# Function to calculate bbox using min-max ranges
def rand_bbox_minmax(img_shape, minmax, count=None):
    """
    Generate random bounding box with min-max constraints.

    Args:
        img_shape (tuple): Shape of the image.
        minmax (tuple): Min and max ratios for bbox.
        count (int): Number of bounding boxes to generate.

    Returns:
        Tuple: Bounding box coordinates.
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


# In[3]:


# Class to handle Mixup and CutMix augmentations
class Mixup:
    """
    Mixup/CutMix implementation that applies different parameters per batch or element.

    Args:
        mixup_alpha (float): Mixup alpha value.
        cutmix_alpha (float): CutMix alpha value.
        cutmix_minmax (List[float]): Min and max image ratios for CutMix.
        prob (float): Probability of applying Mixup or CutMix.
        switch_prob (float): Probability of switching to CutMix.
        mode (str): How to apply Mixup/CutMix ('batch', 'pair', or 'elem').
        correct_lam (bool): Apply lambda correction for CutMix bbox clipping.
        label_smoothing (float): Label smoothing factor.
        num_classes (int): Number of target classes.
    """
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam
        self.mixup_enabled = True  # Set to False to disable during training

    # Add other methods (_params_per_elem, _mix_elem, etc.)


# In[ ]:




