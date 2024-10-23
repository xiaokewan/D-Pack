import xml.etree.ElementTree as ET
import networkx as nx

def parse_net_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    graph = nx.DiGraph()

    def add_blocks(block, parent_name=None):
        block_name = block.get('name')
        if parent_name:
            graph.add_edge(parent_name, block_name)
        for child in block:
            if child.tag == 'block':
                add_blocks(child, block_name)

    add_blocks(root)
    return graph

net_graph = parse_net_file('your_netlist_file.net')
