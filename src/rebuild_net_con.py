import xml.etree.ElementTree as ET
import sys, os, re
from logging import raiseExceptions

from fontTools.unicodedata import block
from param import Boolean


def parse_indexed_signal(signal, full_instance=None):
    """
    Parse signals like:
    - 'alm[7].data_out[3]->LAB_alm_feedback' (explicit instance and port index)
    - 'LAB.data_in[0]' (resolve omitted instance index using full_instance)

    Args:
        signal (str): The signal to parse.
        full_instance (str): The full instance path of the current block, e.g., 'FPGA_packed_netlist[0].LAB[0].alm[0]'.

    Returns:
        tuple: (signal_name, index) where signal_name includes the resolved instance, and index is the port index.
    """
    # Match signals with explicit block instance and port index
    match = re.match(r"(\w+\[\d+\]\.\w+)\[(\d+)\]->", signal)
    if match:
        return match.group(1), int(match.group(2))  # signal_name, index

    # Match signals like 'LAB.data_in[0]' where block instance index might be omitted
    match = re.match(r"(\w+\.\w+)\[(\d+)\]", signal)
    if match:
        base_signal, port_index = match.groups()  # 'LAB.data_in', 0
        block_name, port_name = base_signal.split(".")  # 'LAB', 'data_in'

        # If the block name is missing its index, resolve it from full_instance
        if full_instance:
            # Reverse traverse full_instance to find the correct parent block instance
            parent_blocks = full_instance.split(".")
            for parent_block in reversed(parent_blocks):
                if parent_block.startswith(f"{block_name}["):
                    resolved_signal = f"{parent_block}.{port_name}"  # e.g., 'LAB[0].data_in'
                    return resolved_signal, int(port_index)

    # If no match, return None
    return None, None



def build_block_index(root):
    """
    Build a hierarchical block index where the key is 'parent_instance.instance' and value is the Element.
    """
    block_index = {}

    def index_blocks(block, parent_instance=""):
        instance = block.attrib.get("instance")
        if instance:
            # Construct the hierarchical key
            instance = f"{parent_instance}.{instance}" if parent_instance else instance
            block_index[instance] = block

        # Recursively index child blocks
        for child_block in block.findall("block"):
            index_blocks(child_block, parent_instance=instance if instance else parent_instance)

    # Start indexing from the root
    index_blocks(root)
    return block_index


def find_signal(block_index, full_instance, signal_name, index, isInp):
    """
    Trace signal:
    - If isInp=True, find signal in the current block or parent inputs.
    - If isInp=False, find signal in child blocks (outputs).
    """
    if "." not in signal_name:
        raise ValueError(f"Invalid signal name format: '{signal_name}'. Expected format is '<instance>.<port_name>'.")

    instance, port_name = signal_name.rsplit(".", 1)

    def search_same_layer_or_parent():
        """Search for inputs in the same layer or parent block."""
        # Build key for same layer
        hierarchical_key = construct_full_instance(full_instance, instance)
        target_block = block_index.get(hierarchical_key)

        if target_block is not None:
            for port in target_block.findall("./inputs/port") + target_block.findall("./outputs/port"):
                # Here the inputs can possibly be inputs
                if port.attrib.get("name") == port_name:
                    signals = port.text.strip().split()
                    if index < len(signals):
                        return signals[index]
        return None

    def search_in_child_blocks():
        """Search for outputs in child blocks."""
        current_block = block_index.get(full_instance)
        if current_block is not None:
            for child_block in current_block.findall("./block"):
                if child_block.attrib.get("instance") == instance:
                    for output_port in child_block.findall("./outputs/port"):
                        if output_port.attrib.get("name") == port_name:
                            signals = output_port.text.strip().split()
                            if index < len(signals):
                                return signals[index]
        return None

    # Choose search strategy
    if isInp:  # For inputs
        result = search_same_layer_or_parent()
        print(
            f"      block instance: {full_instance}, sigal name: {signal_name, index}, Input: {isInp}, result: {result}")
    else:  # For outputs
        result = search_in_child_blocks()
        print(
            f"      block instance: {full_instance}, sigal name: {signal_name, index}, Input: {isInp}, result: {result}")
    # TODO: the isInp Is not updating correctly!
    if full_instance == "FPGA_packed_netlist[0].LAB[0].alm[6].lut[1]":
        print("shit")
    return result if result else "open"


def construct_full_instance(full_instance, instance):
    """
    Construct the hierarchical key by appending the instance to the full_instance.

    Args:
        full_instance (str): The full instance path of the current block.
        instance (str): The instance to be appended.

    Returns:
        str: The hierarchical key combining `full_instance` and `instance`.
    """
    # Get (e.g., "LAB" from "LAB[0]")
    current_base_type = full_instance.split(".")[-1].split("[")[0]
    # Get (e.g., "alm" from "alm[9]")
    instance_base_type = instance.split("[")[0]

    if instance_base_type in full_instance:
        # Find where the instance exists in the hierarchy and trim everything after it
        parts = full_instance.split(".")
        for i in range(len(parts)):
            if parts[i].startswith(instance_base_type + "["):
                return ".".join(parts[:i+1])  # Keep up to the found parent instance

    # Check if the instance is a child or sibling
    if current_base_type != instance_base_type:
        # If it's a different type, directly append it as a child
        hierarchical_key = f"{full_instance}.{instance}"
    else:
        # If it's the same type (e.g., sibling), replace the last segment
        parent_path = ".".join(full_instance.split(".")[:-1])
        hierarchical_key = f"{parent_path}.{instance}"

    return hierarchical_key


def resolve_signal_recursive(sig, block_index, visited, full_instance, is_input):
    """
    Recursively resolves a signal to its final endpoint.
    Updates all signals along the path to a unified name.

    Args:
        sig (str): The signal to resolve.
        block_index (dict): The index mapping of blocks.
        visited (set): Tracks visited signals to avoid circular references.
        full_instance (str): The full hierarchical instance path of the current block.
        is_input (bool): Whether the signal is an input or output.

    Returns:
        str: The unified name of the resolved signal.
    """
    if sig == "open":
        return "open"

    # Parse the signal name and index
    # TODO: update here, fix the issue will miss the case "LAB.data_in[0]"
    signal_name, index = parse_indexed_signal(sig, full_instance)

    sig = f"{signal_name}[{index}]" if index is not None else sig
    if sig in visited:
        raise ValueError(f"Circular reference detected for signal: {sig}")

    # Return the resolved signal if already mapped
    # TODO: port mapping should mbuild with full-instance names
    if f"{full_instance}.{sig}" in port_mapping:
        return port_mapping[f"{full_instance}.{sig}"]

    visited.add(f"{full_instance}.{signal_name}[{index}]") if index is not None else visited.add(f"{full_instance}.{sig}")
    if not signal_name:
        # TODO: In this parse, the signal like LAB_datain[0] will be skipped
        return sig  # Return the original signal if parsing fails

    # Search for the resolved signal in the block index
    if is_input:
        resolved_signal = find_signal(block_index, full_instance, signal_name, index, True)
    else:
        resolved_signal = find_signal(block_index, full_instance, signal_name, index, False)

    # If the resolved signal points to another signal, continue the recursion
    instance, port_name = signal_name.rsplit(".", 1)
    if resolved_signal and resolved_signal not in ["open", sig]:
        final_signal = resolve_signal_recursive(resolved_signal, block_index, visited, construct_full_instance(full_instance, instance), is_input)
        # print(f"resolved signal: {resolved_signal}, full_instance: {full_instance}.{instance}, full_instance_constructed: {construct_full_instance(full_instance, instance)}")
    else:
        final_signal = resolved_signal

    # Map all signals along the path to the final resolved signal
    port_mapping[f"{full_instance}.{sig}"] = final_signal
    visited.remove(f"{full_instance}.{signal_name}[{index}]")  # Remove the signal from visited to allow other paths to process
    return final_signal


def update_block_ports_recursive(block, block_index, parent_instance=""):
    """
    Updates the input and output ports of a block recursively by resolving signals.

    Args:
        block (Element): The XML block to process.
        block_index (dict): The index mapping of blocks.
        parent_instance (str): The hierarchical path of the parent block.
    """
    # Construct the full hierarchical instance path
    instance = block.attrib.get("instance", "")
    full_instance = f"{parent_instance}.{instance}" if parent_instance and instance else instance

    # Update input ports
    for input_port in block.findall("./inputs/port"):
        if input_port.text:
            signals = input_port.text.strip().split()
            updated_signals = []
            visited = set()  # Prevent circular references
            for signal in signals:
                resolved_signal = resolve_signal_recursive(signal, block_index, visited, full_instance, True)
                updated_signals.append(resolved_signal)
            input_port.text = " ".join(updated_signals)

    # Update output ports
    for output_port in block.findall("./outputs/port"):
        if output_port.text:
            signals = output_port.text.strip().split()
            updated_signals = []
            visited = set()  # Prevent circular references
            for signal in signals:
                resolved_signal = resolve_signal_recursive(signal, block_index, visited, full_instance, False)
                updated_signals.append(resolved_signal)
            output_port.text = " ".join(updated_signals)

    # Recursively process child blocks
    for child_block in block.findall("block"):
        update_block_ports_recursive(child_block, block_index, full_instance)


def process_xml(file_path, output_folder):
    """Main function"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    block_index = build_block_index(root)
    global port_mapping
    port_mapping = {}
    # update_block_ports(root, block_index)
    update_block_ports_recursive(root, block_index)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(file_path) + '_rebuild.xml')
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        net_file = sys.argv[1]
        output_path = "."
    elif len(sys.argv) != 3:
        print("Usage: python3 rebuild_net_con.py <net_file> <output_path>")
        sys.exit(1)
    else:
        net_file = sys.argv[1]
        output_path = sys.argv[2]
    process_xml(net_file, output_path)


# def update_block_ports(block, block_index, parent_instance=""):
#     """Update block inputs and outputs."""
#
#     # Build full instance path for the current block
#     instance = block.attrib.get("instance", "")
#     full_instance = f"{parent_instance}.{instance}" if parent_instance and instance else instance
#
#     # 1. Process inputs: Search in the same layer or parent block
#     for input_port in block.findall("./inputs/port"):
#         if input_port.text:
#             signals = input_port.text.strip().split()
#             updated_signals = []
#             for signal in signals:
#                 signal_name, index = parse_indexed_signal(signal)
#                 # signal_name = f"{parent_instance}.{??}"
#
#                 if signal_name:
#                     # Extract parent path (excluding current instance)
#                     parent_path = ".".join(full_instance.split(".")[:-1])
#                     combined_signal_name = f"{parent_path}.{signal_name}"
#                 else:
#                     combined_signal_name = None
#
#                 if signal_name is None and "->" in signal:
#                     # Handle signals without explicit instance index
#                     base_signal = signal.split("->")[0]
#                     base_instance, port_with_index = base_signal.split(".")
#                     port_name, index = re.match(r"(\w+)\[(\d+)\]", port_with_index).groups()
#                     combined_signal_name = f"{parent_instance}.{port_name}"
#                     index = int(index)
#
#
#                 if combined_signal_name:
#                     actual_signal = find_signal(block_index, full_instance, combined_signal_name, index, True)
#                     updated_signals.append(actual_signal)
#                 else:
#                     updated_signals.append(signal)
#             input_port.text = " ".join(updated_signals)
#
#     # 2. Process outputs: Search in child blocks
#     for output_port in block.findall("./outputs/port"):
#         if output_port.text:
#             signals = output_port.text.strip().split()
#             updated_signals = []
#             for signal in signals:
#                 signal_name, index = parse_indexed_signal(signal)
#                 if signal_name:
#                     resolved_signal = find_signal(block_index, full_instance, signal_name, index, False)
#                     updated_signals.append(resolved_signal)
#                 else:
#                     updated_signals.append(signal)
#             output_port.text = " ".join(updated_signals)
#
#     # 3. Recursive call for child blocks
#     for child_block in block.findall("block"):
#         update_block_ports(child_block, block_index, full_instance)

