#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/12/25
# @Author  : Xiaoke Wang
# @Group   : UGent HES
# @File    : partition_net_md2.py
# @Software: PyCharm, Ghent


import os
import json
import xml.etree.ElementTree as ET


def preprocess_net_name(net_name):
    """Preprocess net name to remove 'out:' prefix and filter 'open' ports."""
    if net_name.startswith("out:"):
        net_name = net_name[4:]  # Remove 'out:' prefix
    return net_name if net_name and net_name != "open" else None


def is_terminal(block):
    """Check if a block is a terminal (IO, pad, inpad, outpad)."""
    instance = block.get("instance", "").lower()
    mode = block.get("mode", "").lower()
    return any(prefix in instance for prefix in ["io", "pad", "inpad", "outpad", "io_cell"]) or mode in {"io", "inpad",
                                                                                                         "outpad"}


def count_luts_ffs(block):
    """Counts the number of LUTs and FFs inside a given block."""
    lut_count, ff_count = 0, 0

    def traverse(node):
        nonlocal lut_count, ff_count
        for child in node.findall("block"):
            mode = child.get("mode", "").lower()
            if "lut6" in mode:
                lut_count += 1
            if "latch" in mode:
                ff_count += 1
            traverse(child)

    traverse(block)
    return lut_count + ff_count


def partition_hierarchically(block, rent_data=[], depth=0):
    """Recursively partition the circuit based on the XML hierarchy."""

    # Ensure there is a level entry in rent_data
    if len(rent_data) <= depth:
        rent_data.append([])

    # Collect inputs, outputs, and clocks (to count terminals)
    terminals = set()
    for io in block.findall('inputs') + block.findall('outputs') + block.findall('clocks'):
        net_names = [port.text.strip() for port in io.findall('port') if port.text] or [
            io.text.strip() if io.text else None]
        for net_name in net_names:
            if net_name:
                processed_nets = [preprocess_net_name(n) for n in net_name.split()]
                terminals.update(filter(None, processed_nets))  # Remove None values

    # Compute weight of the current block
    weight = count_luts_ffs(block)

    # If the block has no weight, do not record it
    if weight > 0:
        rent_data[depth].append([weight, len(terminals)])

    # Process hierarchical sub-blocks recursively
    for child in block.findall("block"):
        if is_terminal(child):
            continue  # Skip deeply embedded IOs
        partition_hierarchically(child, rent_data, depth + 1)

    return rent_data


# def partition_hierarchically(block, depth=0):
#     """Recursively partition the circuit based on the XML hierarchy."""
#     partitions = []
#
#     # Collect inputs, outputs, and clocks
#     terminals = set()
#     for io in block.findall('inputs') + block.findall('outputs') + block.findall('clocks'):
#         net_names = [port.text.strip() for port in io.findall('port') if port.text] or [
#             io.text.strip() if io.text else None]
#         for net_name in net_names:
#             if net_name:
#                 processed_nets = [preprocess_net_name(n) for n in net_name.split()]
#                 terminals.update(filter(None, processed_nets))
#
#     # Process hierarchical blocks
#     sub_blocks = []
#     for child in block.findall("block"):
#         if is_terminal(child):
#             continue  # Skip deeply embedded IOs
#
#         weight = count_luts_ffs(child)
#         if weight > 0:
#             sub_blocks.append({
#                 "instance": child.get("instance"),
#                 "mode": child.get("mode"),
#                 "weight": weight,
#                 "depth": depth,
#                 "sub_partitions": partition_hierarchically(child, depth + 1)
#             })
#
#     partitions.append({
#         "instance": block.get("instance"),
#         "mode": block.get("mode"),
#         "level": depth,
#         "terminals": list(terminals),
#         "weight": count_luts_ffs(block),
#         "blocks": sub_blocks
#     })
#     return partitions


def process_net_file(net_file, output_folder):
    """Process the netlist file and extract hierarchical partitions."""
    os.makedirs(output_folder, exist_ok=True)
    tree = ET.parse(net_file)
    root = tree.getroot()

    partitioning = partition_hierarchically(root)

    output_path = os.path.join(output_folder, os.path.basename(net_file) + '.hierarchical.json')
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(partitioning, fp, indent=4)
    print(f"Partitioning data saved to {output_path}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 partition_xml.py <net_file> <output_path>")
        sys.exit(1)
    net_file = sys.argv[1]
    output_path = sys.argv[2]
    process_net_file(net_file, output_path)
