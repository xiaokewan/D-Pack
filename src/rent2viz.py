#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/20/24
# @Author  : Marieke Louage, Xiaoke Wang
# @Group   : UGent HES
# @File    : rent2viz.py.py
# @Software: PyCharm, Ghent
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys


def trend_line(data, slope_threshold=(0.2, 1)):
    filtered_data = []
    for i in range(1, len(data)):
        slope = (data[i, 1] - data[i - 1, 1]) / (data[i, 0] - data[i - 1, 0]) if (data[i, 0] - data[
            i - 1, 0]) != 0 else 0
        if slope_threshold[0] <= slope <= slope_threshold[1]:
            filtered_data.append(data[i - 1])
            filtered_data.append(data[i])

    if not filtered_data:
        return None, None, None, None

    # filtered data for tendline
    filtered_data = np.array(filtered_data)
    x, y = filtered_data[:, 0], filtered_data[:, 1]
    x_mean, y_mean = x.mean(), y.mean()
    x_err, y_err = x - x_mean, y - y_mean
    a = (x_err * y_err).sum() / (x_err ** 2).sum()
    b = y_mean - a * x_mean
    error = np.sum((y - (a * x + b)) ** 2) / len(filtered_data)

    return np.array([[x[0], a * x[0] + b], [x[-1], a * x[-1] + b]]), a, b, error


def visualize_rent(rent_path, output_filename='Rents_rule_real.png', output_figures_folder="."):
    if not rent_path.endswith('.json'):
        raise ValueError(f"Expected a .json file, got {rent_path} instead.")
    with open(rent_path, "r", encoding="utf-8") as fp:  # Load JSON
        rent_data = json.load(fp)

    # Flatten data
    rent_data_flat = np.array([point for level in rent_data for point in level])
    blocks, pins = rent_data_flat[:, 0], rent_data_flat[:, 1]
    rent_data_flat = rent_data_flat[:, 0:2]

    # Bin data
    n_bins = len(rent_data)
    max_blocks = blocks.max()
    bin_factor = max_blocks ** (1 / n_bins)
    bin_values = np.round(bin_factor ** np.arange(1, n_bins + 1))
    bin_values[-1] += 1  # Ensure covering max value

    # Mean and median per bin
    bin_means = []
    for i in range(n_bins):
        bin_mask = (blocks <= bin_values[i]) if i == 0 else ((blocks > bin_values[i - 1]) & (blocks <= bin_values[i]))
        bin_data = rent_data_flat[bin_mask]
        if bin_data.size > 0:
            blocks_mean = bin_data[:, 0].mean()
            pins_median = np.median(bin_data[:, 1])
            bin_means.append([blocks_mean, pins_median])

    bin_means = np.array(bin_means)
    log_bin_means = np.log2(bin_means)
    line, slope, _, _ = trend_line(log_bin_means)
    plt.figure(figsize=(10, 6))
    plt.scatter(blocks, pins, alpha=0.1, label='Data Points')
    plt.scatter(bin_means[:, 0], bin_means[:, 1], s=100, color='red', alpha=0.85, edgecolors='w', linewidths=2,
                marker='o', label='Bin Means')
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel('$B$ (Blocks)', size=20)
    plt.ylabel('$T$ (Terminals)', size=20)
    if line is not None:
        plt.plot(np.exp2(line[:, 0]), np.exp2(line[:, 1]), color='black', linewidth=2, linestyle='--',
                 label=f'Slope (r) = {slope:.2f}')
    else:
        print("Warning: No valid trend line found, skipping trend line plot.")

    # plt.title('Rent\'s Rule Visualization')
    plt.legend(fontsize=14, loc='lower right')

    os.makedirs(output_figures_folder, exist_ok=True)
    plt.savefig(os.path.join(output_figures_folder, output_filename), format='pdf')
    # plt.show()
    print(f"Rents' Exponent is: {slope}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 rent2viz.py <json_file_path>  <output_figures_folder>")
        sys.exit(1)

    rent_file_path = sys.argv[1]
    output_figures_folder = sys.argv[2]
    output_filename = os.path.basename(rent_file_path) + "_viz.pdf"

    visualize_rent(rent_file_path, output_filename, output_figures_folder)
    print(f"Visualization saved to {output_filename}")
