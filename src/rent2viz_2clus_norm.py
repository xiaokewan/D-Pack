#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/12/25
# @Author  : Xiaoke Wang
# @Group   : UGent HES
# @File    : rent2viz_recursive.py
# @Software: PyCharm, Ghent

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys


def rent_norm(t_dic, r):
    weighted_blocks = []
    for bl_dic in t_dic:
        w = 0
        for vertice in bl_dic:
            w += (bl_dic[vertice]) * (int(vertice) ** (1 / r))
        weighted_blocks.append(w)
    return np.array(weighted_blocks)


def trend_line_recursive(bin_means, t_dic, iterations=20):
    """Recursive computation of Rent's exponent."""
    _, slope, _, _ = trend_line(bin_means)  # Initial observed slope
    norm_blocks = bin_means[:, 0]  # Initialize norm_blocks for output tracking

    for i in range(iterations):
        prev_slope = slope
        norm_blocks = rent_norm(t_dic, slope)
        log_bin_means = np.column_stack((np.log2(norm_blocks), bin_means[:, 1]))  # Only apply log2 to norm_blocks
        line, slope, _, _ = trend_line(log_bin_means)

        if prev_slope is not None and abs(prev_slope - slope) <= 0.01:
            break

    return slope, line, norm_blocks, log_bin_means


def trend_line(data, slope_threshold=(0.2, 1)):
    """Compute a trendline for the given dataset with a slope filter."""
    x, y = data[:, 0], data[:, 1]
    x_mean, y_mean = x.mean(), y.mean()
    a = np.cov(x, y)[0, 1] / np.var(x)
    b = y_mean - a * x_mean
    error = np.mean((y - (a * x + b)) ** 2)
    return np.array([[x[0], a * x[0] + b], [x[-1], a * x[-1] + b]]), a, b, error


def load_rent_data(rent_path):
    """Load Rent's rule data from a JSON file."""
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
        rent_data = all_levels

    rent_data_flat = np.array([point[:2] for level in rent_data for point in level])
    t_dic = [point[2] if len(point) > 2 else {} for level in rent_data for point in level]

    return rent_data_flat, t_dic


def visualize_rent_dual(file1, file2, output_filename="Rents_rule_dual_recursive.pdf", output_figures_folder="."):
    """Visualize Rent's rule using recursive weighted normalization."""
    rent_data_1, t_dic_1 = load_rent_data(file1)
    rent_data_2, t_dic_2 = load_rent_data(file2)

    bin_means_1 = np.log2(rent_data_1)
    bin_means_2 = np.log2(rent_data_2)

    slope_1, line_1, norm_blocks_1, log_bin_means_1 = trend_line_recursive(bin_means_1, t_dic_1)
    slope_2, line_2, norm_blocks_2, log_bin_means_2 = trend_line_recursive(bin_means_2, t_dic_2)

    plt.figure(figsize=(10, 6))
    plt.scatter(norm_blocks_1, np.exp2(bin_means_1[:, 1]), alpha=0.1, color='blue', label='Norm Inter-CLB Data')
    plt.scatter(norm_blocks_2, np.exp2(bin_means_2[:, 1]), alpha=0.1, color='green', label='Norm Intra-CLB Data')
    plt.plot(np.exp2(line_2[:, 0]), np.exp2(line_2[:, 1]), '--', color='brown', label=f'Trend (Inter-CLB) r = {slope_2:.4f}')
    plt.plot(np.exp2(line_1[:, 0]), np.exp2(line_1[:, 1]), '--', color='black', label=f'Trend (Intra-CLB) r = {slope_1:.4f}')

    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel('$B_{norm}$', size=20)
    plt.ylabel('$T$', size=20)
    plt.legend(fontsize=18, loc='lower right')

    os.makedirs(output_figures_folder, exist_ok=True)
    plt.savefig(os.path.join(output_figures_folder, output_filename), format='pdf')
    plt.show()
    print(f"Saved visualization to {output_filename}")
    print(f"Inter-CLB exponent: {slope_1:.4f}, Intra-CLB exponent: {slope_2:.4f}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 rent2viz_recursive.py <json_file_1> <json_file_2> <output_folder>")
        sys.exit(1)
    visualize_rent_dual(sys.argv[1], sys.argv[2], output_figures_folder=sys.argv[3])
