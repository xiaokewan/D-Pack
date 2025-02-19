import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re


def parse_vpr_log(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    patterns = {
        "cpd": r"Final critical path delay \(least slack\): ([\d.]+) ns",
        "twl": r"Total wirelength: (\d+),",
        "packing_time": r"# Packing took ([\d.]+) seconds",
        "placement_time": r"# Placement took ([\d.]+) seconds",
        "routing_time": r"# Routing took ([\d.]+) seconds",
        "total_time": r"The entire flow of VPR took ([\d.]+) seconds",
        "LAB_usage": r"Netlist\s+.*?(\d+)\s+blocks of type: LAB",
        "LAB_utilization": r"Physical Tile LAB:\s*\n\s*Block Utilization: ([\d.]+) Logical Block: LAB",
        "channel_width": r"Circuit successfully routed with a channel width factor of (\d+)",
        "fpga_size": r"FPGA sized to (\d+) x (\d+) \(auto\)"  # Extracts two values (width & height)
    }

    extracted_values = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.DOTALL)
        if match:
            if key == "fpga_size":  # Handle FPGA size separately
                width = int(match.group(1))
                height = int(match.group(2))
                extracted_values["fpga_area"] = width * height  # Calculate area
            else:
                extracted_values[key] = float(match.group(1)) if "." in match.group(1) else int(match.group(1))

    return extracted_values


# Function to parse packing_pin_util.rpt for actual pin usage
def parse_packing_pin_util(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    # Extract LAB Input and Output averages
    lab_input_avg_match = re.search(r"Type: LAB\n.*?Input Pin Usage:\n.*?Avg: ([\d.]+)", content, re.DOTALL)
    lab_output_avg_match = re.search(r"Type: LAB\n.*?Output Pin Usage:\n.*?Avg: ([\d.]+)", content, re.DOTALL)

    lab_input_avg = float(lab_input_avg_match.group(1)) if lab_input_avg_match else None
    lab_output_avg = float(lab_output_avg_match.group(1)) if lab_output_avg_match else None

    # Compute actual pin usage
    if lab_input_avg is not None and lab_output_avg is not None:
        actual_pin_usage = (lab_input_avg + lab_output_avg) / 131
    else:
        actual_pin_usage = None

    return {"actual_pin_usage": actual_pin_usage}


# Function to process all VPR logs in directories named by util (0.0 - 1.0)
def process_vpr_logs(base_dir):
    results = []

    # Filter only numeric directories
    valid_dirs = [d for d in os.listdir(base_dir) if re.match(r"^\d+(\.\d+)?$", d)]

    # Sort numerically
    for dir_name in sorted(valid_dirs, key=float):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):  # Ensure it's a directory
            log_file = os.path.join(dir_path, "vpr_stdout.log")
            packing_util_file = os.path.join(dir_path, "packing_pin_util.rpt")

            if os.path.exists(log_file) and os.path.exists(packing_util_file):
                print(f"Processing: {log_file} & {packing_util_file}")

                parsed_data = parse_vpr_log(log_file)
                parsed_data.update(parse_packing_pin_util(packing_util_file))
                parsed_data["util"] = float(dir_name)  # Add utilization value

                results.append(parsed_data)

    return results


# Function to save results to Excel
def save_to_excel(data, output_path):
    if not data:
        print("No data to save.")
        return
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
    print(f"Saved results to {output_path}")


# Function to plot results with log scaling for better visibility
def plot_results(data):
    if not data:
        print("No data available for plotting.")
        return

    df = pd.DataFrame(data)

    # Plot each metric on a logarithmic scale
    metrics = ["cpd", "total_wirelength", "packing_time", "placement_time", "routing_time", "total_time", "actual_pin_usage",
               "lab_usage", "lab_utilization", "channel_width"]

    plt.figure(figsize=(10, 6))
    for metric in metrics:
        if metric in df.columns:
            plt.plot(df["util"], df[metric], marker='o', label=metric)

    plt.xlabel("Target Ext Pin Utilization")
    plt.ylabel("Metric Value (log scale)")
    plt.yscale("log")  # Logarithmic scale for better visibility
    plt.title("VPR Metrics vs. Target Ext Pin Utilization")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()



# Main function
def main():
    # Default values
    default_input_dir = "/media/xiaokewan/TOSHIBA/Code_phd/D-Pack/gnl_example/stratixiv/1000_0.8"
    default_output_file = "../results/gnl_example/stratixiv/1000_0.8/vpr_results.xlsx"

    # Get command-line arguments
    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input_dir
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output_file

    print(f"Using input directory: {input_dir}")
    print(f"Results will be saved in: {output_file}")

    results = process_vpr_logs(input_dir)
    save_to_excel(results, output_file)
    plot_results(results)


if __name__ == "__main__":
    main()
