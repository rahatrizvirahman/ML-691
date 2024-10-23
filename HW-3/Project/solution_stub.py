#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@updated_by: Rahat Rizvi Rahman
"""

# Required packages:
# pip install torch torchvision matplotlib numpy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np



def my_train(train_dataset, test_dataset = None):
    W = np.random.randn(10,784)
    b = np.random.randn(10)
    # above is random, your code should produce "trained" W & b
    return W, b

def my_test(W,b,test_dataset):
    test_error_rate = np.random.randn();
    # above is random, your code should produce actual error rate (the fraction of misclassified samples from test_dataset)
    return test_error_rate


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    def find_zero_pixels(train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
        all_zeros = None

        for data, _ in train_loader:
            data = data.view(-1, 28 * 28)
            batch_zeros = (data == 0).all(dim=0)
            if all_zeros is None:
                all_zeros = batch_zeros
            else:
                all_zeros &= batch_zeros

        return all_zeros.numpy()


    def analyze_zero_overlap(W, zero_pixels, zero_threshold = 1e-2):
        W[np.abs(W) <= zero_threshold] = 0.0
        W_zeros = (W == 0).all(axis=0)
        
        both_zeros = np.logical_and(W_zeros, zero_pixels)
        only_image_zeros = np.logical_and(zero_pixels, np.logical_not(W_zeros))
        only_W_zeros = np.logical_and(W_zeros, np.logical_not(zero_pixels))
        
        return {
            'both': np.sum(both_zeros),
            'only_images': np.sum(only_image_zeros),
            'only_W': np.sum(only_W_zeros)
        }, W_zeros

    def plot_binary_pattern(zero_pixels,title):
        plt.figure(figsize=(6, 6))
        plt.imshow(zero_pixels.reshape(28, 28), cmap='binary')
        plt.title(title)
        plt.colorbar()
        plt.show()

    
        
    def create_custom_colormap():
        # Create a continuous red to navy colormap
        colors = [
            (0.6, 0, 0),      # Dark red
            (1, 0, 0),        # Red
            (1, 0.5, 0.5),    # Light red
            (1, 0.7, 0.7),    # Very light red / Pink
            (1, 0.85, 0.85),  # Very light pink
            (1, 1, 1),        # White
            (0.85, 0.85, 1),  # Very light blue
            (0.7, 0.7, 1),    # Light blue
            (0.5, 0.5, 1),    # Medium blue
            (0, 0, 1),        # Blue
            (0, 0, 0.6)       # Navy
        ]
        n_bins = 256  # Increased for smoother transition
        cmap_full = LinearSegmentedColormap.from_list("custom_full", colors, N=n_bins)
        
        # Create a colormap with black for zero and full spectrum for non-zero
        colors_with_black = np.vstack((np.array([0, 0, 0, 1]), cmap_full(np.linspace(0, 1, 255))))
        cmap_with_black = ListedColormap(colors_with_black)
        
        return cmap_with_black

    def plot_weight_images(W):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        cmap = create_custom_colormap()
        
        for i in range(10):
            ax = axes[i]
            w = W[i].reshape(28, 28)
            
            # Normalize the data to [-1, 1] range
            abs_max = np.abs(w).max()
            w_norm = w / abs_max
            
            # Shift the normalized data to [0, 1] range, keeping 0 at 0
            w_shifted = (w_norm + 1) / 2
            w_shifted[w == 0] = 0  # Ensure exact zeros stay at 0
            
            # Plot the image
            im = ax.imshow(w_shifted, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f'Class {i}')
            ax.axis('off')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cbar.set_ticklabels([f'-{abs_max:.2f}', f'-{abs_max/2:.2f}', '0', f'{abs_max/2:.2f}', f'{abs_max:.2f}'])
        
        fig.suptitle('Pixel weights in trained W for each class', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()


    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_dataset_raw = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    # Find zero pixels in the training dataset
    mnist_zero_pixels = find_zero_pixels(train_dataset_raw)
    print(f"Number of zero pixels in all training images: {np.sum(mnist_zero_pixels)}")

    # Train the model and get W and b
    print("Training the model...")
    W, b = my_train(train_dataset, test_dataset = test_dataset)



    # Test the model
    test_error_rate = my_test(W, b, test_dataset)
    print(f"Test error rate: {test_error_rate:.4f}")

    # Test the model
    Wtrunc = np.copy(W);
    Wtrunc[np.abs(Wtrunc)<1e-2]=0.0
    test_error_rate = my_test(Wtrunc, b, test_dataset)
    print(f"Test error rate (tiny Ws zeroed out): {test_error_rate:.4f}")

    # Analyze the overlap between zero pixels and zeros in W
    overlap_results, W_zero_pixels = analyze_zero_overlap(W, mnist_zero_pixels, zero_threshold = 1e-2)
    print(f"Zeros in both images and W: {overlap_results['both']}")
    print(f"Zeros only in images: {overlap_results['only_images']}")
    print(f"Zeros only in W: {overlap_results['only_W']}")


    mnist_zero_pixels = find_zero_pixels(train_dataset_raw)
    overlap_results, W_zero_pixels = analyze_zero_overlap(W, mnist_zero_pixels, zero_threshold = 1e-2)

    num_zeros_ovelap = overlap_results['both'];
    num_zeros_Wonly = overlap_results['only_W'];
    num_zeros_total = num_zeros_ovelap + num_zeros_Wonly;
        
    
    print(num_zeros_ovelap, num_zeros_Wonly, num_zeros_total)
        # Plot binary pattern
    plot_binary_pattern(mnist_zero_pixels,title="0-in-all-imgs/non0-in-at-least-one-img, over 50k training imgs")

    plot_binary_pattern(W_zero_pixels,title="0-in-all-10-classes/non0-in-at-least-one-class, over trained W")

    # Plot weight images
    plot_weight_images(W)

