import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys


def detect_segments(data):
    """Automatically detect slope transition points to segment the data."""
    slopes = np.diff(np.log2(data[:, 1])) / np.diff(np.log2(data[:, 0]))
    mean_slope = np.mean(slopes)

    # Identify transition points: when slope deviates significantly from the mean
    transition_indices = np.where(np.abs(slopes - mean_slope) > 0.35)[0] + 1  # +1 to get the right-side index

    # Ensure three segments: before, within, and after the high-slope region
    if len(transition_indices) < 2:
        return [0, len(data) // 2, len(data)]  # Fallback to default split

    return [0] + list(transition_indices) + [len(data)]


def trend_line(data):
    """Compute a least-squares trend line for the given data segment."""
    if len(data) < 2:
        return None, None, None, None

    x, y = np.log2(data[:, 0]), np.log2(data[:, 1])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    line = np.array([[x[0], slope * x[0] + intercept], [x[-1], slope * x[-1] + intercept]])

    return line, slope, intercept, np.sum((y - (slope * x + intercept)) ** 2) / len(y)


def visualize_rent(rent_path, output_filename='Rents_rule_real.png', output_figures_folder="."):
    if not rent_path.endswith('.json'):
        raise ValueError(f"Expected a .json file, got {rent_path} instead.")

    with open(rent_path, "r", encoding="utf-8") as fp:
        rent_data = json.load(fp)

    rent_data_flat = np.array([point for level in rent_data for point in level])
    blocks, pins = rent_data_flat[:, 0], rent_data_flat[:, 1]

    n_bins = len(rent_data)
    bin_factor = blocks.max() ** (1 / n_bins)
    bin_values = np.round(bin_factor ** np.arange(1, n_bins + 1))
    bin_values[-1] += 1

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

    # **NEW**: Detect slope transition points
    segment_indices = detect_segments(bin_means)

    plt.figure(figsize=(10, 6))
    plt.scatter(blocks, pins, alpha=0.1, label='Data Points')
    plt.scatter(bin_means[:, 0], bin_means[:, 1], s=100, color='red', alpha=0.85, edgecolors='w', linewidths=2,
                marker='o', label='Bin Means')

    import itertools
    colors = itertools.cycle(['blue', 'green', 'black', 'purple', 'orange', 'cyan'])

    for i in range(len(segment_indices) - 1):
        start, end = segment_indices[i], segment_indices[i + 1]
        segment = bin_means[start:end]

        line, slope, _, _ = trend_line(segment)
        if line is not None:
            plt.plot(np.exp2(line[:, 0]), np.exp2(line[:, 1]), color=next(colors), linewidth=2, linestyle='--',
                     label=f'Trend {i + 1}: Slope = {slope:.2f}')

    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel('$B$ (Blocks)', size=15)
    plt.ylabel('$T$ (Terminals)', size=15)
    plt.title("Rent's Rule with Multi-Segment Trend Lines")
    plt.legend()

    os.makedirs(output_figures_folder, exist_ok=True)
    plt.savefig(os.path.join(output_figures_folder, output_filename))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 rent2viz.py <json_file_path> <output_figures_folder>")
        sys.exit(1)

    rent_file_path = sys.argv[1]
    output_figures_folder = sys.argv[2]
    output_filename = os.path.basename(rent_file_path) + "_viz_seg.png"

    visualize_rent(rent_file_path, output_filename, output_figures_folder)
    print(f"Visualization saved to {output_filename}")

