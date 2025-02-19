#!/bin/bash

# Define paths
VPR_PATH="/home/xiaokewan/Software/vtr-verilog-to-routing-master/vpr/vpr"
ARCH_FILE="/home/xiaokewan/Software/vtr-verilog-to-routing-master/vtr_flow/arch/titan/stratixiv_arch.timing.xml"

# Check if BLIF file is provided or if -h is passed
if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 <path/to/blif_file>"
    echo "Runs VPR with the specified BLIF file and sweeps target_ext_pin_util from 0.1 to 0.9."
    exit 1
fi


# Get full path of input BLIF file
BLIF_FILE_PATH=$(realpath "$1")

# Extract base directory and file name
BASE_DIR=$(dirname "$BLIF_FILE_PATH")
BLIF_FILE=$(basename "$BLIF_FILE_PATH")

# Store original directory
ORIGINAL_DIR=$(pwd)

# Sweep target_ext_pin_util from [0.0, 1.0] (from dense to loose)
for UTIL in $(awk 'BEGIN{for(i=0.00; i<=0.30; i+=0.04) printf "%.2f ", i}'); do
    # Create a directory for each UTIL value
    RUN_DIR="${BASE_DIR}/${UTIL}"
    mkdir -p "$RUN_DIR"

    # Change to the run directory
    cd "$RUN_DIR" || exit 1

    # Run VPR inside the directory (VPR will generate its own logs)
    echo "----------------------------------------------------"
    echo "Running VPR with target_ext_pin_util=${UTIL} in ${RUN_DIR}..."
    echo "----------------------------------------------------"
    $VPR_PATH "$ARCH_FILE" "$BLIF_FILE_PATH" --target_ext_pin_util "$UTIL" --route_chan_width 44 > "vpr_log.txt" 2>&1

    # Return to the original directory
    cd "$ORIGINAL_DIR" || exit 1
done

echo "Sweeping completed!"

