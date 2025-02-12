import os, re, subprocess, sys, json
import numpy as np
import xml.etree.ElementTree as ET


def _preprocess_net_name(net_name):
    """Preprocess net name to remove 'out:' prefix and filter 'open' ports."""
    if net_name.startswith("out:"):
        net_name = net_name[4:]  # Remove 'out:' prefix
    return net_name if net_name and net_name != "open" else None


class Hypergraph:
    def __init__(self, hypergraph, external_edges, weights, folder, name='hg', suffix=''):
        """Initialize the hypergraph and filter valid edges."""
        self.hypergraph = np.array([self._filter_and_deduplicate_edges(edges) for edges in hypergraph], dtype=object)

        valid_edges = {edge for edges in self.hypergraph for edge in edges}
        self.external_edges = np.array([
            _preprocess_net_name(e) for e in external_edges if e and e in valid_edges
        ])

        self.n_vertices = len(self.hypergraph)
        self.n_pins = len(self.external_edges)
        self.folder = folder
        self.name_base = name
        self.suffix = suffix
        self.weights = weights

    def _filter_and_deduplicate_edges(self, edges):
        """Remove 'open' ports and deduplicate edges within each block."""
        return list({_preprocess_net_name(edge) for edge in edges if edge and edge != "open"})

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

        hmetis_lines = [f"{len(edge_nodes)} {self.n_vertices} 10"]
        hmetis_lines.extend(" ".join(nodes) for nodes in edge_nodes.values())

        for vertex in range(0, self.n_vertices):
            weight = self.weights.get(vertex, 1)
            hmetis_lines.append(str(weight))

        with open(self.get_path_graphfile(), 'w') as file1:
            file1.writelines([entry + '\n' for entry in hmetis_lines])

    def run_hmetis(self, hmetis_path):
        """Run hMETIS partitioning."""
        subprocess.run([hmetis_path, self.get_path_graphfile(), '2', '5', '10', '1', '1', '2', '1', '1'],
                       capture_output=True)

    def split(self, hmetis_path):
        """Perform bipartitioning using hMETIS."""
        self.print_hmetis_hypergraph()
        self.run_hmetis(hmetis_path)

        path_splitfile = self.get_path_graphfile() + '.part.2'

        # **Check if hMetis output file exists**
        if not os.path.exists(path_splitfile):
            raise FileNotFoundError(f"Partition file {path_splitfile} missing. Check hMetis execution.")

        # **Read hMetis output**
        with open(path_splitfile, 'r') as file1:
            mask = np.array(list(map(int, file1.readlines())))

        # **Ensure mask length matches number of vertices**
        if len(mask) != self.n_vertices:
            raise ValueError(
                f"  ERROR:Partition mask length {len(mask)} does not match the number of vertices {self.n_vertices}.")

        hypergraph0 = [self.hypergraph[i] for i in range(len(mask)) if mask[i] == 0]
        hypergraph1 = [self.hypergraph[i] for i in range(len(mask)) if mask[i] == 1]

        if len(hypergraph0) == 0 or len(hypergraph1) == 0:
            print(f"️   Warning: One of the partitions is empty! Returning single partition.")
            return self, None

        cut_edges = np.array(list(set(np.concatenate(hypergraph0)) & set(np.concatenate(hypergraph1))))
        self.external_edges = np.unique(np.append(self.external_edges, cut_edges))

        # **Fix weights mapping: use enumerate(mask) instead of self.weights.keys()**
        weights0 = {i + 1: self.weights.get(i + 1, 1) for i, part in enumerate(mask) if part == 0}
        weights1 = {i + 1: self.weights.get(i + 1, 1) for i, part in enumerate(mask) if part == 1}

        return Hypergraph(hypergraph0, self.external_edges, weights0, self.folder, self.name_base, '0'), \
            Hypergraph(hypergraph1, self.external_edges, weights1, self.folder, self.name_base, '1')

    def get_path_graphfile(self):
        return os.path.join(self.folder, self.name_base + self.suffix)


def preprocess_net_name(net_name):
    """Preprocess net name to remove 'out:' prefix and filter 'open' ports."""
    if net_name.startswith("out:"):
        net_name = net_name[4:]  # Remove 'out:' prefix
    return net_name if net_name and net_name != "open" else None


def parse_net_file_to_hypergraph(file_path, output_folder):
    """Parse the netlist XML into a hypergraph structure with correct block weights."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    hypergraph_data = []
    external_edges = set()
    block_weights = {}

    def is_embedded_block(block):
        """Check if a block is deeply embedded and should be ignored."""
        instance = block.get("instance", "").lower()
        mode = block.get("mode", "").lower()

        return any(re.match(r"^(io|pad|inpad|outpad|io_cell)\[\d+\]$", instance) for instance in [instance]) \
            or mode in {"io", "inpad", "outpad"}

    def count_basic_blocks(block):
        """Counts the number of LUTs and FFs inside the given block."""
        lut_count = 0
        ff_count = 0
        # Recursively count LUTs and FFs inside child blocks
        def traverse_children(node):
            nonlocal lut_count, ff_count
            for child in node.findall("block"):
                mode = child.get("mode", "").lower()
                if "lut6" in mode:
                    lut_count += 1
                if "latch" in mode:
                    ff_count += 1
                traverse_children(child)
        traverse_children(block)

        if lut_count == 0 and ff_count == 0:
            if block.get("mode", "").lower() == "io" and block.get("name", "").lower() != "open":
                print(
                    f"    Block {block.get('instance')} weighted 1 ")
                return 1
        else:
            print(
                f"    Block {block.get('instance')} weighted {lut_count + ff_count}")
            return lut_count + ff_count

    def add_blocks(block, depth=0):
        """Recursively process the netlist and extract hypergraph edges and block weights."""
        if is_embedded_block(block) and depth > 1:
            return  # Skip deeply embedded blocks

        edges = []
        # Collect inputs, outputs, and clocks
        for io in block.findall('inputs') + block.findall('outputs') + block.findall('clocks'):
            net_names = [port.text.strip() for port in io.findall('port') if port.text] or [
                io.text.strip() if io.text else None]
            for net_name in net_names:
                if net_name:
                    filtered_nets = [preprocess_net_name(name) for name in net_name.split()]
                    filtered_nets = [net for net in filtered_nets if net]  # Remove None values
                    edges.extend(filtered_nets)

                    if block == root:
                        external_edges.update(filtered_nets)

        # Add to hypergraph if not empty
        if edges:
            weight = count_basic_blocks(block)
            if weight is not None and weight > 0:
                hypergraph_data.append(edges)
                block_weights[len(hypergraph_data)] = weight

        # Recursively process child blocks
        for child_block in block.findall('block'):
            add_blocks(child_block, depth + 1)

    add_blocks(root)

    return Hypergraph(hypergraph_data, list(external_edges), block_weights, output_folder)


def bipartition(hg, rent_data, hmetis_path, partition_level=0):
    """Recursively bipartition the hypergraph, tracking weighted Rent's Rule data."""

    # **Use sum of block weights instead of counting vertices**
    weighted_blocks = sum(hg.weights.values())  # Consider block weights
    pins = hg.n_pins  # Terminal count remains the same
    blocks = hg.n_vertices
    # **Store weighted blocks in rent_data**
    if len(rent_data) <= partition_level:
        rent_data.append([[weighted_blocks, pins]])
    else:
        rent_data[partition_level].append([weighted_blocks, pins])

    if blocks > 2:
        hg0, hg1 = hg.split(hmetis_path)
        if hg0 is None or len(hg0.hypergraph) == 0:
            print(f"    Alert: Skipping empty partition at level {partition_level}")
            return
        if hg1 is None or len(hg1.hypergraph) == 0:
            print(f"    Alert: Skipping empty partition at level {partition_level}")
            return

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