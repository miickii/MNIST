import matplotlib.pyplot as plt
import numpy as np
import torch

def show_image_and_target(images, targets, show=True):
    """
    Display a grid of images with their corresponding labels.

    Args:
        images (torch.Tensor or np.ndarray): Batch of images to display (shape: [N, H, W] or [N, 1, H, W]).
        targets (torch.Tensor or np.ndarray): Corresponding labels for the images (shape: [N]).
        show (bool): If True, display the plot. Otherwise, save the plot to a file.
    """
    # Ensure images and targets are numpy arrays
    if isinstance(images, torch.Tensor):
        images = images.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    # Handle grayscale images with shape [N, 1, H, W]
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0, :, :]  # Remove the channel dimension

    num_images = images.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_images)))  # Create a square grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f"Label: {targets[i]}")
        ax.axis('off')

    plt.tight_layout()
    if show:
        plt.show()
