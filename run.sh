#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./run.sh <input_net_file>"
    exit 1
fi

# Define input and output directories
INPUT_NET_FILE=$1
BASE_NAME=$(basename "$INPUT_NET_FILE" .net)
OUTPUT_DIR="./results/${BASE_NAME}/"
REBUILD_NET_FILE="${OUTPUT_DIR}${BASE_NAME}.net_rebuild.xml"
RENT_JSON_FILE="${REBUILD_NET_FILE}.hierarchical.json"
HMETIS_PATH="./hmetis/hmetis"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Step 1: Run rebuild_net_con.py
echo "Running rebuild_net_con.py..."
python3 ./src/rebuild_net_con.py "$INPUT_NET_FILE" "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "Error running rebuild_net_con.py"
    exit 1
fi
echo "Finished rebuild_net_con."

# Step 2: Run partition_net.py
echo "Running partition_net.py..."
python3 ./src/partition_net_md2.py "$REBUILD_NET_FILE" "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "Error running partition_net.py"
    exit 1
fi
echo "Finished Partitioning."
# Step 3: Run rent2viz.py
echo "Running rent2viz.py..."
python3 ./src/rent2viz.py "$RENT_JSON_FILE" "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo "Error running rent2viz.py"
    exit 1
fi

echo "Pipeline completed successfully!"
