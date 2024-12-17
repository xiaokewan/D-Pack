import xml.etree.ElementTree as ET
import sys, os, re

from param import Boolean


def parse_indexed_signal(signal):
    """parse signals like: alm[7].data_out[3]->LAB_alm_feedback """
    match = re.match(r"(\w+\[\d+\]\.\w+)\[(\d+)\]->", signal)
    if match:
        return match.group(1), int(match.group(2))  # signal_name, index
    return None, None


def build_block_index(root):
    """block index"""
    # TODO： good for one layer before, not sure there is exceptions.
    block_index = {}

    def index_blocks(block):
        instance = block.attrib.get("instance")
        if instance:
            block_index[instance] = block
        for child_block in block.findall("block"):
            index_blocks(child_block)

    index_blocks(root)
    return block_index


def find_signal(block_index, parent_block, signal_name, index, current_block, isInp):
    """
    Trace signal:
    1. If isInp=True, find signal in the same layer or parent block (inputs).
    2. If isInp=False, find signal in child blocks (outputs).
    """

    if "." not in signal_name:  # Signal name without hierarchy
        return "open"

    instance, port_name = signal_name.split(".")
    base_instance = instance.split("[")[0]  # e.g., "alm" from "alm[7]"

    def find_in_inputs():
        """Search for signal in same layer or parent block."""
        # Search in the same layer (outputs of sibling blocks)
        target_block = block_index.get(instance)
        if target_block is not None:
            for output_port in target_block.findall("./inputs/port"):
                if output_port.attrib.get("name") == port_name:
                    signals = output_port.text.strip().split()
                    if index < len(signals):
                        return signals[index]

        # Search in parent block inputs
        if parent_block is not None:
            for input_port in parent_block.findall("./inputs/port"):
                if input_port.attrib.get("name") == base_instance:
                    signals = input_port.text.strip().split()
                    if index < len(signals):
                        return signals[index]
        return None

    def find_in_outputs():
        """Search for signal in child blocks."""
        for child_block in current_block.findall("./block"):
            if child_block.attrib.get("instance") == instance:
                for output_port in child_block.findall("./outputs/port"):
                    if output_port.attrib.get("name") == port_name:
                        signals = output_port.text.strip().split()
                        if index < len(signals):
                            return signals[index]
        return None

    # Determine search strategy based on `isInp`
    if isInp:  # Input-related signal
        result = find_in_inputs()
    else:  # Output-related signal
        result = find_in_outputs()

    return result if result else "open"


def update_block_ports(block, block_index, parent_block=None):
    """Update block inputs and outputs."""

    # 1. Process inputs: Search in same layer or parent block
    for input_port in block.findall("./inputs/port"):
        if input_port.text:
            signals = input_port.text.strip().split()
            updated_signals = []
            for signal in signals:
                signal_name, index = parse_indexed_signal(signal)

                if signal_name is None and "->" in signal:
                    # The case for uptrace the signal without index
                    base_signal = signal.split("->")[0]
                    base_instance, port_with_index = base_signal.split(".")

                    if parent_block is not None and base_instance == parent_block.attrib.get("mode"):
                        parent_instance = parent_block.attrib.get("instance", "")
                        port_name, index = re.match(r"(\w+)\[(\d+)\]", port_with_index).groups()
                        signal_name = f"{parent_instance}.{port_name}"
                        index = int(index)

                if signal_name:
                    actual_signal = find_signal(block_index, parent_block, signal_name, index, block, True)
                    updated_signals.append(actual_signal)
                else:
                    updated_signals.append(signal)
            input_port.text = " ".join(updated_signals)

    # 2. Process outputs: Search in child blocks
    for output_port in block.findall("./outputs/port"):
        if output_port.text:
            signals = output_port.text.strip().split()
            updated_signals = []
            for signal in signals:
                signal_name, index = parse_indexed_signal(signal)

                if signal_name:
                    resolved_signal = find_signal(block_index, None, signal_name, index, block, False)
                    updated_signals.append(resolved_signal)
                else:
                    updated_signals.append(signal)
            output_port.text = " ".join(updated_signals)

    # 3. Recursive call for child blocks
    for child_block in block.findall("block"):
        update_block_ports(child_block, block_index, block)


def process_xml(file_path, output_folder):
    """Main function"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    block_index = build_block_index(root)
    update_block_ports(root, block_index)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(file_path) + '_rebuild.xml')
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        net_file = sys.argv[1]
        output_path = "."
    elif len(sys.argv) != 3:
        print("Usage: python3 rebuild_net_con.py <net_file>  <output_path>")
        sys.exit(1)
    else:
        net_file = sys.argv[1]
        output_path = sys.argv[2]
    process_xml(net_file, output_path)
