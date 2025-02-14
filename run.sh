#!/bin/bash

# Usage message
usage() {
    echo "Usage: ./run.sh -m <mode> <input_net_file>"
    echo "  -m <mode> : Choose partitioning method (default: 2)"
    echo "     0 : Use partition_net.py (default, single JSON, rent2viz.py)"
    echo "     1 : Use partition_net_md2.py (hierarchical, single JSON, rent2viz.py)"
    echo "     2 : Use partition_net_md3.py (INTRA/INTER, two JSONs, rent2viz_2clus.py)"
    echo "  <input_net_file> : The input netlist file"
    exit 1
}

# Default mode
MODE=2

# Parse arguments
while getopts "m:" opt; do
    case ${opt} in
        m) MODE=${OPTARG} ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

# Check if input file is provided
if [ "$#" -ne 1 ]; then
    usage
fi

# Define input and output directories
INPUT_NET_FILE=$1
BASE_NAME=$(basename "$INPUT_NET_FILE" .net)
RELATIVE_DIR=$(dirname "$INPUT_NET_FILE")  # Get relative path (e.g., gnl_example/stratixiv/10000)
OUTPUT_DIR="./results/${RELATIVE_DIR}/${BASE_NAME}/"

REBUILD_NET_FILE="${OUTPUT_DIR}${BASE_NAME}.net_rebuild.xml"
RENT_JSON_FILE="${REBUILD_NET_FILE}.hierarchical.json"   # Used for mode 0 and 1
INTER_JSON_FILE="${OUTPUT_DIR}inter_LAB_partition.json"  # Used for mode 2
INTRA_JSON_FILE="${OUTPUT_DIR}intra_LAB_partition.json"  # Used for mode 2
HMETIS_PATH="./hmetis/hmetis"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Step 1: Run rebuild_net_con.py
echo -e "\n========== 🏗️ Running Netlist Reconstruction =========="
echo "🔹 Input file: $INPUT_NET_FILE"
echo "🔹 Output directory: $OUTPUT_DIR"
echo "🔹 Running: rebuild_net_con.py..."
python3 ./src/rebuild_net_con.py "$INPUT_NET_FILE" "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo -e "❌ Error: rebuild_net_con.py failed!"
    exit 1
fi
echo -e "✔️ Finished rebuild_net_con.\n"

# Step 2: Run the chosen partitioning method
echo "========== 🧩 Running Partitioning =========="
PARTITION_SCRIPT="partition_net.py"  # Default
RENT_VIZ_SCRIPT="rent2viz.py"        # Default for -m 0 and -m 1
RENT_VIZ_INPUT="$RENT_JSON_FILE"     # Default for -m 0 and -m 1

if [ "$MODE" -eq 0 ]; then
    echo "🔹 Mode: Standard partitioning (partition_net.py)"
elif [ "$MODE" -eq 1 ]; then
    PARTITION_SCRIPT="partition_net_md2.py"
    echo "🔹 Mode: Hierarchical partitioning (partition_net_md2.py)"
elif [ "$MODE" -eq 2 ]; then
    PARTITION_SCRIPT="partition_net_md3.py"
    RENT_VIZ_SCRIPT="rent2viz_2clus.py"  # Uses two JSONs
    RENT_VIZ_INPUT="$INTRA_JSON_FILE $INTER_JSON_FILE"
    echo "🔹 Mode: INTRA/INTER partitioning (partition_net_md3.py)"
fi

echo "🔹 Running: $PARTITION_SCRIPT..."
python3 "./src/$PARTITION_SCRIPT" "$REBUILD_NET_FILE" "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo -e "❌ Error: $PARTITION_SCRIPT failed!"
    exit 1
fi
echo -e "✔️ Finished Partitioning.\n"

# Step 3: Run the correct rent visualization script
echo "========== 📊 Running Rent's Rule Visualization =========="
echo "🔹 Visualization script: $RENT_VIZ_SCRIPT"
python3 "./src/$RENT_VIZ_SCRIPT" $RENT_VIZ_INPUT "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo -e "❌ Error: $RENT_VIZ_SCRIPT failed!"
    exit 1
fi

# Show final output locations
echo -e "\n========== ✅ Pipeline Completed Successfully! =========="
echo -e "📂 Results saved in:"
echo -e "\t🔹 Netlist Rebuild File: $REBUILD_NET_FILE"
if [ "$MODE" -eq 2 ]; then
    echo -e "\t🔹 INTRA-LAB JSON: $INTRA_JSON_FILE"
    echo -e "\t🔹 INTER-LAB JSON: $INTER_JSON_FILE"
else
    echo -e "\t🔹 Rent JSON File: $RENT_JSON_FILE"
fi
echo -e "\t📊 Visualization Output: Saved in $OUTPUT_DIR\n"
