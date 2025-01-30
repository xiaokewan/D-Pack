import os
import xml.etree.ElementTree as ET
import numpy as np
import json
import subprocess
import sys


class Hypergraph:
    def __init__(self, hypergraph, external_edges, folder, name='hg', suffix=''):
        self.hypergraph = np.array(hypergraph, dtype=object)
        self.external_edges = np.array(external_edges)

        # Filter only valid external edges
        external_edges_mask = np.zeros(len(self.external_edges), dtype=bool)
        for edges in hypergraph:
            for edge in edges:
                indices = np.argwhere(self.external_edges == edge)
                if indices.size > 0:
                    external_edges_mask[indices[0][0]] = True
        self.external_edges = self.external_edges[external_edges_mask]

        self.n_vertices = len(self.hypergraph)
        self.n_pins = len(self.external_edges)
        self.folder = folder
        self.name_base = name
        self.suffix = suffix

    def print_hmetis_hypergraph(self):
        """Generate hypergraph format for hMETIS partitioning."""
        edge_nodes = {}
        for i, edges in enumerate(self.hypergraph):
            node = str(i + 1)
            for edge in edges:
                if edge in edge_nodes:
                    edge_nodes[edge].append(node)
                else:
                    edge_nodes[edge] = [node]

        hmetis_lines = [f"{len(edge_nodes)} {self.n_vertices}"]
        hmetis_lines.extend(" ".join(nodes) for nodes in edge_nodes.values())

        with open(self.get_path_graphfile(), 'w') as file1:
            file1.writelines([entry + '\n' for entry in hmetis_lines])

    def run_hmetis(self, hmetis_path):
        """Run hMETIS partitioning."""
        subprocess.run([hmetis_path, self.get_path_graphfile(), '2', '5', '10', '1', '1', '2', '1', '0'],
                       capture_output=True)

    def split(self, hmetis_path):
        """Perform bipartitioning using hMETIS."""
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
    """Parse the netlist XML into a hypergraph structure."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    hypergraph_data = []
    external_edges = set()  # Use a set for uniqueness

    def add_blocks(block):
        edges = []
        # Collect inputs, outputs, and clocks
        for io in block.findall('inputs') + block.findall('outputs') + block.findall('clocks'):
            net_names = [port.text.strip() for port in io.findall('port') if port.text] or [io.text.strip() if io.text else None]
            for net_name in net_names:
                if net_name:
                    edges.extend(net_name.split())
                    if block == root:
                        external_edges.update(net_name.split())

        # Add to hypergraph if not empty
        if edges:
            hypergraph_data.append(edges)

        # Recursively process child blocks
        for child_block in block.findall('block'):
            add_blocks(child_block)

    add_blocks(root)

    return Hypergraph(hypergraph_data, list(external_edges), output_folder)


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