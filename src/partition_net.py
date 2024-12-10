import os
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import json
import subprocess
import sys

from astropy.units.quantity_helper.function_helpers import block


class Hypergraph:
    def __init__(self, hypergraph, external_edges, folder, name='hg', suffix=''):
        self.hypergraph = np.array(hypergraph, dtype=object)
        self.external_edges = np.array(external_edges)

        # filter
        external_edges_mask = np.zeros(len(self.external_edges))
        for edges in hypergraph:
            for edge in edges:
                indices = np.argwhere(self.external_edges == edge)
                if indices.size > 0:
                    index = indices[0]
                    external_edges_mask[index] = 1
        self.external_edges = self.external_edges[external_edges_mask == 1]

        self.n_vertices = len(self.hypergraph)
        self.n_pins = len(self.external_edges)
        self.folder = folder
        self.name_base = name
        self.suffix = suffix

    def print_hmetis_hypergraph(self):
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

        hmetis_lines = [f'{self.n_hyperedges} {self.n_vertices}']
        hmetis_lines.extend(' '.join(nodes) for nodes in edge_nodes.values())
        with open(self.get_path_graphfile(), 'w') as file1:
            file1.writelines([entry + '\n' for entry in hmetis_lines])

    def run_hmetis(self, hmetis_path):
        output = subprocess.run([hmetis_path, self.get_path_graphfile(), '2', '5', '10', '1', '1', '2', '1', '0'],
                                capture_output=True)

    def split(self, hmetis_path):
        self.print_hmetis_hypergraph()
        self.run_hmetis(hmetis_path)
        path_splitfile = self.get_path_graphfile() + '.part.2'
        with open(path_splitfile, 'r') as file1:
            mask = np.array(list(map(int, file1.readlines())))

        hypergraph0 = self.hypergraph[mask == 0]
        hypergraph1 = self.hypergraph[mask == 1]
        cut_edges = np.array(list(set(np.concatenate(hypergraph0)) & set(np.concatenate(hypergraph1))))
        self.external_edges = np.unique(np.append(self.external_edges, cut_edges))

        return Hypergraph(hypergraph0, self.external_edges, self.folder, self.name_base, '0'), \
            Hypergraph(hypergraph1, self.external_edges, self.folder, self.name_base, '1')

    def get_path_graphfile(self):
        return os.path.join(self.folder, self.name_base + self.suffix)


def parse_net_file_to_hypergraph(file_path, output_folder):
    import xml.etree.ElementTree as ET

    tree = ET.parse(file_path)
    root = tree.getroot()
    hypergraph_data = []
    external_edges = []


    top_level_name = root.attrib.get("name", "top")

    def preprocess_net_name(net_name, block_name, block_instance):
        """Preprocess net name to include block hierarchy, but exclude top-level name."""
        base_name = net_name.split('->')[0] if '->' in net_name else net_name
        # Remove top-level block name from hierarchy
        if block_name.startswith(top_level_name):
            block_name = block_name[len(top_level_name) + 1:]  # Strip top-level prefix
        return f"top_{block_instance}.{base_name}" if block_name == "" else f"{block_name}_{block_instance}.{base_name}"

    def add_blocks(block, parent_name=""):
        block_name = block.attrib.get("name", "unknown")
        block_instance = block.attrib.get("instance", "unknown")
        full_block_name = f"{parent_name}-{block_name}" if parent_name != "" else block_name

        edges = []

        # Collect all inputs, outputs, and clocks for the current block
        for io in block.findall('inputs') + block.findall('outputs') + block.findall('clocks'):
            for net in io.iter():
                net_name = net.text.strip() if net.text else None
                if net_name:  # Filter out "open" connections
                    valid_nets = [
                        preprocess_net_name(name, full_block_name, block_instance)
                        for name in net_name.split() if name != "open"
                    ]
                    print(f"valid edges: {len(valid_nets)}")
                    edges.extend(valid_nets)
                    # If it's an external connection, add it to external_edges
                    if block == root:
                        external_edges.extend(valid_nets)

        # Add this block's connections to the hypergraph
        if edges:
            hypergraph_data.append(edges)

        # Recursively process child blocks
        for child_block in block.findall('block'):
            add_blocks(child_block, full_block_name)

    # Start processing from the root block
    add_blocks(root)

    # Remove duplicates in external edges
    external_edges = list(set(external_edges))
    return Hypergraph(hypergraph_data, external_edges, output_folder)


def bipartition(hg, rent_data, hmetis_path, partition_level=0):
    blocks, pins = hg.n_vertices, hg.n_pins
    if len(rent_data) <= partition_level:
        rent_data.append([[blocks, pins]])
    else:
        rent_data[partition_level].append([blocks, pins])

    if blocks > 2:
        hg0, hg1 = hg.split(hmetis_path)
        bipartition(hg0, rent_data, hmetis_path, partition_level + 1)
        bipartition(hg1, rent_data, hmetis_path, partition_level + 1)


def process_net_file(net_file, output_folder, hmetis_path):
    os.makedirs(output_folder, exist_ok=True)
    hypergraph = parse_net_file_to_hypergraph(net_file, output_folder)
    rent_data = []
    bipartition(hypergraph, rent_data, hmetis_path)
    output_path = os.path.join(output_folder, os.path.basename(net_file) + '.rent')

    ## json or pickle
    with open(output_path + '.json', "w", encoding="utf-8") as fp:
        json.dump(rent_data, fp, indent=4)
    # with open(output_path, "wb") as fp:
    #     pickle.dump(rent_data, fp)
    print(f"Rent data saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) == 2:
        net_file = sys.argv[1]
        output_path = "."
    elif len(sys.argv) != 3:
        print("Usage: python3 partition_net.py <net_file>  <output_path>")
        sys.exit(1)
    else:
        net_file = sys.argv[1]
        output_path = sys.argv[2]
    hmetis_path = './hmetis/hmetis'

    process_net_file(net_file, output_path, hmetis_path)