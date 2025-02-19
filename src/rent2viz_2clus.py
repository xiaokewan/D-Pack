#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/12/25 
# @Author  : Xiaoke Wang
# @Group   : UGent HES
# @File    : rent2viz_2clus.py
# @Software: PyCharm, Ghent
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys


def trend_line(data, slope_threshold=(0.2, 1)):
    """Compute a trendline for the given dataset with a slope filter."""
    filtered_data = []
    for i in range(1, len(data)):
        slope = (data[i, 1] - data[i - 1, 1]) / (data[i, 0] - data[i - 1, 0]) if (data[i, 0] - data[i - 1, 0]) != 0 else 0
        if slope_threshold[0] <= slope <= slope_threshold[1]:
            filtered_data.append(data[i - 1])
            filtered_data.append(data[i])

    if not filtered_data:
        return None, None, None, None

    filtered_data = np.array(filtered_data)
    x, y = filtered_data[:, 0], filtered_data[:, 1]
    x_mean, y_mean = x.mean(), y.mean()
    x_err, y_err = x - x_mean, y - y_mean
    a = (x_err * y_err).sum() / (x_err ** 2).sum()
    b = y_mean - a * x_mean
    error = np.sum((y - (a * x + b)) ** 2) / len(filtered_data)

    return np.array([[x[0], a * x[0] + b], [x[-1], a * x[-1] + b]]), a, b, error


def load_rent_data(rent_path):
    """Load JSON Rent's rule data from file and return flattened data."""
    if not rent_path.endswith('.json'):
        raise ValueError(f"Expected a .json file, got {rent_path} instead.")
    with open(rent_path, "r", encoding="utf-8") as fp:
        rent_data = json.load(fp)

    if isinstance(rent_data, dict):
        all_levels = []
        for lab_name, partitions in rent_data.items():
            if isinstance(partitions, list):  # Ensure valid format
                for level_idx, level_data in enumerate(partitions):
                    while len(all_levels) <= level_idx:
                        all_levels.append([])  # Ensure level exists
                    all_levels[level_idx].extend(level_data)
        rent_data = all_levels  # Replace with correctly structured format

    # Flatten data and extract `[blocks, terminals]` pairs
    rent_data_flat = np.array([point for level in rent_data for point in level])
    blocks, pins = rent_data_flat[:, 0], rent_data_flat[:, 1]

    return rent_data_flat[:, 0:2], blocks, pins, len(rent_data)


def visualize_rent_dual(file1, file2, output_filename="Rents_rule_dual.png", output_figures_folder="."):
    """Visualize Rent's rule data for two JSON files in different colors."""

    # Load both datasets
    rent_data_1, blocks_1, pins_1, n_bins_1 = load_rent_data(file1)
    rent_data_2, blocks_2, pins_2, n_bins_2 = load_rent_data(file2)

    # Bin data for both datasets
    def bin_rent_data(rent_data, blocks, n_bins):
        max_blocks = blocks.max()
        bin_factor = max_blocks ** (1 / n_bins)
        bin_values = np.round(bin_factor ** np.arange(1, n_bins + 1))
        bin_values[-1] += 1  # Ensure covering max value

        bin_means = []
        for i in range(n_bins):
            bin_mask = (blocks <= bin_values[i]) if i == 0 else ((blocks > bin_values[i - 1]) & (blocks <= bin_values[i]))
            bin_data = rent_data[bin_mask]
            if bin_data.size > 0:
                blocks_mean = bin_data[:, 0].mean()
                pins_median = np.median(bin_data[:, 1])
                bin_means.append([blocks_mean, pins_median])
        return np.array(bin_means)

    bin_means_1 = bin_rent_data(rent_data_1, blocks_1, n_bins_1)
    bin_means_2 = bin_rent_data(rent_data_2, blocks_2, n_bins_2)

    log_bin_means_1 = np.log2(bin_means_1)
    log_bin_means_2 = np.log2(bin_means_2)

    line_1, slope_1, _, _ = trend_line(log_bin_means_1)
    line_2, slope_2, _, _ = trend_line(log_bin_means_2)

    # Plot settings
    plt.figure(figsize=(10, 6))
    plt.scatter(blocks_1, pins_1, alpha=0.1, color='blue', label='Data Points (Inter-CLB)')
    plt.scatter(blocks_2, pins_2, alpha=0.1, color='green', label='Data Points (Intra-CLB)')
    plt.scatter(bin_means_1[:, 0], bin_means_1[:, 1], s=100, color='red', alpha=0.85, edgecolors='w', linewidths=2,
                marker='o', label='Bin Means (Inter-CLB)')
    plt.scatter(bin_means_2[:, 0], bin_means_2[:, 1], s=100, color='orange', alpha=0.85, edgecolors='w', linewidths=2,
                marker='s', label='Bin Means (Intra-CLB)')

    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel('$B$', size=20)
    plt.ylabel('$T$', size=20)

    # Plot trendlines if found
    if line_1 is not None:
        plt.plot(np.exp2(line_1[:, 0]), np.exp2(line_1[:, 1]), color='black', linewidth=2, linestyle='--',
                 label=f'Trend (Inter-CLB) r = {slope_1:.4f}')
    else:
        print("Warning: No valid trend line found for Inter-CLB.")

    if line_2 is not None:
        plt.plot(np.exp2(line_2[:, 0]), np.exp2(line_2[:, 1]), color='brown', linewidth=2, linestyle='--',
                 label=f'Trend (Intra-CLB) r = {slope_2:.4f}')
    else:
        print("Warning: No valid trend line found for Intra-CLB.")

    # plt.title('Rent\'s Rule Visualization (Intra-CLB vs. Inter-CLB)')
    plt.legend(fontsize=18, loc='lower right')

    os.makedirs(output_figures_folder, exist_ok=True)
    plt.savefig(os.path.join(output_figures_folder, output_filename), format='pdf')
    # plt.show()
    print(f"    Inter-LAB: {slope_1:.4f},\n  Intra-LAB: {slope_2:.4f}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 rent2viz_dual.py <json_file_1> <json_file_2> <output_figures_folder>")
        sys.exit(1)

    rent_file_1 = sys.argv[1]
    rent_file_2 = sys.argv[2]
    output_figures_folder = sys.argv[3]

    output_filename = f"Rents_rule_dual_{os.path.basename(rent_file_1)}_{os.path.basename(rent_file_2)}.pdf"

    visualize_rent_dual(rent_file_1, rent_file_2, output_filename, output_figures_folder)
    print(f"Visualization saved to {output_filename}")
