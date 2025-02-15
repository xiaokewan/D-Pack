#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/8/25 
# @Author  : Xiaoke Wang
# @Group   : UGent HES
# @File    : blif2rent.py.py
# @Software: PyCharm, Ghent

import os
import json
import numpy as np
import subprocess
import csv
import sys


class Hypergraph:
    def __init__(self, hypergraph, external_edges, folder, name='hg', suffix=''):
        # Extract hypergraph nodename = list index + 1 (starts from 1), list of edgenames per node
        self.hypergraph = np.array(hypergraph, dtype=object)

        # Input and output edges are external edges
        self.external_edges = np.array(external_edges)

        # Exclude external edges that are not connected to a node in the hypergraph
        external_edges_mask = np.zeros(len(self.external_edges))
        for edges in hypergraph:
            for edge in edges:
                indices = np.argwhere(self.external_edges == edge)
                if indices.size > 0:
                    index = np.argwhere(self.external_edges == edge)[0]
                    external_edges_mask[index] = 1

        self.external_edges = self.external_edges[external_edges_mask == 1]

        self.n_vertices = len(self.hypergraph)  ## Blocks
        self.n_pins = len(self.external_edges)  ## Pins

        self.folder = folder
        self.name_base = name
        self.suffix = suffix

    def print_hmetis_hypergraph(self):
        # Hmetis expects flipped graph (edge -> nodes) instead of (node -> edges)
        edge_nodes = {}
        for i, edges in enumerate(self.hypergraph):
            node = str(i + 1)
            for edge in edges:
                if edge in edge_nodes:
                    edge_nodes[edge].append(node)
                else:
                    edge_nodes[edge] = [node]
        self.n_hyperedges = len(edge_nodes)
        self.n_vertices = len(self.hypergraph)
        # Print file
        hmetis_lines = []
        hmetis_lines.append(' '.join([str(self.n_hyperedges), str(self.n_vertices)]))
        for nodes in edge_nodes.values():
            hmetis_lines.append(' '.join(nodes))
        file1 = open(self.get_path_graphfile(), 'w')
        lines = file1.writelines([entry + '\n' for entry in hmetis_lines])
        file1.close()

    def run_hmetis(self, hmetis_path):
        output = subprocess.run([hmetis_path, self.get_path_graphfile(), '2', '5', '10', '1', '1', '2', '1', '0'],
                                capture_output=True)

    def split(self, hmetis_path):
        # make input file for hmetis
        self.print_hmetis_hypergraph()
        # run hmetis
        self.run_hmetis(hmetis_path)
        # process output file (name inputfile + '.part.2') --> format read hmetis docs
        path_splitfile = self.get_path_graphfile() + '.part.2'
        file1 = open(path_splitfile, 'r')
        lines = file1.readlines()
        file1.close()

        # split hypergraph nodes
        mask = np.array(list(map(int, lines)))  # partition1: 0, partition2: 1
        hypergraph0 = self.hypergraph[mask == 0]
        hypergraph1 = self.hypergraph[mask == 1]

        # add cut edges to external edges
        cut_edges = np.array(list(set(np.concatenate(hypergraph0)) & set(np.concatenate(hypergraph1))))
        self.external_edges = np.unique(np.append(self.external_edges, cut_edges))

        hg0 = Hypergraph(self.hypergraph[mask == 0], self.external_edges, self.folder, self.name_base, '0')
        hg1 = Hypergraph(self.hypergraph[mask == 1], self.external_edges, self.folder, self.name_base, '1')

        return hg0, hg1

    def get_path_graphfile(self):
        return os.path.join(self.folder, self.name_base + self.suffix)


def get_hypergraph_from_blif(path_blif, path_graphfiles_folder):
    with open(path_blif, 'r') as f:
        lines = f.readlines()

    models = []
    index1 = 0
    for index2, line in enumerate(lines):
        if '.model' in line:
            models.append(lines[index1:index2])
            index1 = index2
    models.append(lines[index1:])
    main_model = models[1]

    new_main_model = []
    new_line_split = []
    for line in main_model:
        line_split = line.split()
        if line_split and line_split[-1] == '\\':
            new_line_split.extend(line_split[:-1])
        else:
            new_line_split.extend(line_split)
            new_main_model.append(new_line_split)
            new_line_split = []

    inputs, outputs, names, latches, subckts = None, None, [], [], []
    for split_line in new_main_model:
        if split_line:
            if split_line[0] == '.inputs':
                inputs = split_line
            elif split_line[0] == '.outputs':
                outputs = split_line
            elif split_line[0] == '.names':
                names.append(split_line)
            elif split_line[0] == '.latch':
                latches.append(split_line)
            elif split_line[0] == '.subckt':
                subckts.append(split_line)

    hypergraph_internal = []
    for subckt in subckts:
        edges = [io_net.split('=')[1] for io_net in subckt[2:]]
        hypergraph_internal.append(edges)

    for name in names:
        hypergraph_internal.append(name[1:])

    for latch in latches:
        hypergraph_internal.append(latch[1:3])

    external_edges = inputs + outputs
    return Hypergraph(hypergraph_internal, external_edges, path_graphfiles_folder)


def bipartition(hg, rent_data, hmetis_path, partition_level=0):
    blocks, pins = hg.n_vertices, hg.n_pins
    if len(rent_data) > partition_level:
        rent_data[partition_level].append([blocks, pins])
    else:
        rent_data.append([[blocks, pins]])
    if blocks > 2:
        hg0, hg1 = hg.split(hmetis_path)
        bipartition(hg0, rent_data, hmetis_path, partition_level + 1)
        bipartition(hg1, rent_data, hmetis_path, partition_level + 1)


def process_blif_file(blif_file, output_folder, hmetis_path):
    os.makedirs(output_folder, exist_ok=True)
    hypergraph = get_hypergraph_from_blif(blif_file, output_folder)
    rent_data = []
    bipartition(hypergraph, rent_data, hmetis_path)

    json_output_path = os.path.join(output_folder, os.path.basename(blif_file) + '.json')
    csv_output_path = os.path.join(output_folder, os.path.basename(blif_file) + '.csv')

    with open(json_output_path, "w", encoding="utf-8") as fp:
        json.dump(rent_data, fp, indent=4)

    with open(csv_output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for sublist in rent_data:
            writer.writerow(sublist)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 blif2rent.py <blif_file_path> <hmetis_path> <output_folder>")
        sys.exit(1)

    blif_file_path, hmetis_path, output_folder = sys.argv[1], sys.argv[2], sys.argv[3]
    base_name = os.path.basename(blif_file_path)
    base_name = base_name.replace('.blif', '')
    # output_folder = f"./results/{base_name}"
    process_blif_file(blif_file_path, output_folder, hmetis_path)
