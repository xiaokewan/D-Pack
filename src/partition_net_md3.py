import os, sys
import json
import xml.etree.ElementTree as ET
import numpy as np
import subprocess


def _preprocess_net_name(net_name):
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

def is_ff_lut(block):
    """Check if a block is a LUT or FFs."""
    mode = block.get("mode", "").lower()
    return "lut6" in mode or "latch" in mode

def count_luts_ffs(block):
    """Counts the number of LUTs and FFs inside a given block, including itself if applicable."""
    lut_count, ff_count = 0, 0
    def traverse(node):
        nonlocal lut_count, ff_count
        mode = node.get("mode", "").lower()
        if "lut6" in mode:
            lut_count += 1
        if "latch" in mode:
            ff_count += 1
        for child in node.findall("block"):
            traverse(child)
    traverse(block)
    return lut_count + ff_count


def extract_hypergraph(block, lab_only=True):
    """
    Extracts a hypergraph from the given XML block.
    - If `lab_only=True`: Only considers LABs and IOs.
    - If `lab_only=False`: Considers LUTs and FFs inside each LAB.
    """
    hypergraph_data = []
    block_weights = []
    external_edges = set()  # Only updated at root level

    def add_blocks(node, is_root=False):
        edges = []

        # Root Block: Collect external edges (top-level terminals)
        if is_root and (
                (lab_only and "FPGA_packed_netlist" in node.get("instance", "")) or
                (not lab_only and "LAB" in node.get("mode", ""))
        ):
            for io in node.findall('inputs') + node.findall('outputs') + node.findall('clocks'):
                net_names = [port.text.strip() for port in io.findall('port') if port.text] or [
                    io.text.strip() if io.text else None]
                for net_name in net_names:
                    if net_name:
                        processed_nets = [_preprocess_net_name(n) for n in net_name.split()]
                        external_edges.update(filter(None, processed_nets))  # Store as external terminals

        # Non-root Blocks: Collect only internal edges
        elif "FPGA_packed_netlist" not in node.get("instance", "") and not is_terminal(node):
            if lab_only or is_ff_lut(node):
                for io in node.findall('inputs') + node.findall('outputs') + node.findall('clocks'):
                    net_names = [port.text.strip() for port in io.findall('port') if port.text] or [
                        io.text.strip() if io.text else None]
                    for net_name in net_names:
                        if net_name:
                            processed_nets = [_preprocess_net_name(n) for n in net_name.split()]
                            edges.extend(filter(None, processed_nets))  # Internal edges

                if edges:
                    weight = count_luts_ffs(node)  # ✅ Compute weight at any level
                    if weight > 0:
                        hypergraph_data.append(edges)
                        block_weights.append(weight)


        # ✅ Recursively process children
        for child in node.findall("block"):
            if node.get("mode", "").lower() not in ["io", "lab"] or not lab_only:
                if not is_terminal(child):  # Skip deeply embedded IOs
                    add_blocks(child, is_root=False)

    # Start processing with root node
    add_blocks(block, is_root=True)
    return hypergraph_data, list(external_edges), block_weights


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

        for vertex in range(self.n_vertices):
            weight = self.weights[vertex] if vertex < len(self.weights) else 1  # Avoid out-of-bounds error
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
        weights0 = [self.weights[i] for i, part in enumerate(mask) if part == 0]
        weights1 = [self.weights[i] for i, part in enumerate(mask) if part == 1]

        # weights0 = {i : self.weights.get(i + 1, 1) for i, part in enumerate(mask) if part == 0}
        # weights1 = {i : self.weights.get(i + 1, 1) for i, part in enumerate(mask) if part == 1}

        return Hypergraph(hypergraph0, self.external_edges, weights0, self.folder, self.name_base, '0'), \
            Hypergraph(hypergraph1, self.external_edges, weights1, self.folder, self.name_base, '1')

    def get_path_graphfile(self):
        return os.path.join(self.folder, self.name_base + self.suffix)


# def bipartition(hg, rent_data, hmetis_path, partition_level=0):
#     """Recursively bipartition the hypergraph, tracking weighted Rent's Rule data."""
#     weighted_blocks = sum(hg.weights)
#     pins = hg.n_pins
#     blocks = hg.n_vertices
#
#     if len(rent_data) <= partition_level:
#         rent_data.append([])
#     rent_data[partition_level].append([weighted_blocks, pins])
#
#     if blocks > 2:
#         hg0, hg1 = hg.split(hmetis_path)
#         if hg0 is None or len(hg0.hypergraph) == 0:
#             return
#         if hg1 is None or len(hg1.hypergraph) == 0:
#             return
#         del hg
#         bipartition(hg0, rent_data, hmetis_path, partition_level + 1)
#         bipartition(hg1, rent_data, hmetis_path, partition_level + 1)


def bipartition(hg, rent_data, hmetis_path, partition_level=0):
    '''Recursively bipartition the hypergraph, tracking weighted Rent's Rule data.'''
    weighted_blocks = sum(hg.weights)
    pins = hg.n_pins
    blocks = hg.n_vertices

    pin_count_dict = {}
    for vertex_index in range(hg.n_vertices):
        pin_count = len(hg.hypergraph[vertex_index])
        if pin_count in pin_count_dict:
            pin_count_dict[pin_count] += 1
        else:
            pin_count_dict[pin_count] = 1

    if len(rent_data) >= partition_level + 1:
        rent_data[partition_level].append([weighted_blocks, pins, pin_count_dict])
    else:
        rent_data.append([[weighted_blocks, pins, pin_count_dict]])
    if blocks > 2:
        hg0, hg1 = hg.split(hmetis_path)
        del hg
        bipartition(hg0, rent_data, hmetis_path, partition_level + 1)
        bipartition(hg1, rent_data, hmetis_path, partition_level + 1)


def process_net_file(net_file, output_folder, hmetis_path):
    os.makedirs(output_folder, exist_ok=True)
    tree = ET.parse(net_file)
    root = tree.getroot()

    # **Step 1: Partition LAB-Level**
    print("Partitioning LAB-Level Hypergraph...")
    hypergraph_data, external_edges, weights = extract_hypergraph(root, lab_only=True)
    lab_hypergraph = Hypergraph(hypergraph_data, external_edges, weights, output_folder, "1intra_LAB")

    rent_data = []
    bipartition(lab_hypergraph, rent_data, hmetis_path)
    lab_output_path = os.path.join(output_folder, "inter_LAB_partition.json")
    with open(lab_output_path, "w", encoding="utf-8") as fp:
        json.dump(rent_data, fp, indent=4)
    print(f"LAB-Level Partitioning saved to {lab_output_path}")

    # **Step 2: Partition Inter-LAB (Inside each LAB)**
    print("Partitioning Inter-LAB Hypergraphs...")
    intra_lab_data = {}
    for lab_block in root.findall("block"):
        if is_terminal(lab_block):
            continue
        sub_hypergraph_data, sub_external_edges, sub_weights = extract_hypergraph(lab_block, lab_only=False)
        if len(sub_hypergraph_data) == 0:
            continue
        lab_name = lab_block.get("instance", "LAB")
        intra_lab_graph = Hypergraph(sub_hypergraph_data, sub_external_edges, sub_weights, output_folder, f"2intra_LAB")

        intra_rent_data = []
        bipartition(intra_lab_graph, intra_rent_data, hmetis_path)
        intra_lab_data[lab_name] = intra_rent_data

    intra_lab_output_path = os.path.join(output_folder, "intra_LAB_partition.json")
    with open(intra_lab_output_path, "w", encoding="utf-8") as fp:
        json.dump(intra_lab_data, fp, indent=4)
    print(f"Inter-LAB Partitioning saved to {intra_lab_output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 partition_net_hybrid.py <net_file> <output_path>")
        sys.exit(1)
    net_file = sys.argv[1]
    output_path = sys.argv[2]
    hmetis_path = './hmetis/hmetis'
    process_net_file(net_file, output_path, hmetis_path)
