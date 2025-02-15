#!/bin/bash

# Exit immediately on error
set -e

# Usage function
usage() {
    echo "Usage: $0 <blif_file>"
    exit 1
}

# Check if a file is provided
if [ "$#" -ne 1 ]; then
    usage
fi

# Define input and output directories
BLIF_FILE=$1
BASE_NAME=$(basename "$BLIF_FILE" .blif)
RELATIVE_DIR=$(dirname "$BLIF_FILE")  # Get relative path (e.g., gnl_example/stratixiv/10000)
OUTPUT_DIR="./results/${RELATIVE_DIR}/${BASE_NAME}/"

RENT_JSON_FILE="${OUTPUT_DIR}${BASE_NAME}.blif.json"
HMETIS_PATH="./hmetis/hmetis"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Display processing information
echo -e "\n========== 🏗️ Processing BLIF File =========="
echo "🔹 Input file: $BLIF_FILE"
echo "🔹 Base name: $BASE_NAME"
echo "🔹 Output directory: $OUTPUT_DIR"
echo "🔹 Rent JSON file: $RENT_JSON_FILE"

# Step 1: Convert BLIF to Rent's Rule JSON
echo -e "\n========== 🔄 Running blif2rent.py =========="
python3 ./src/blif2rent.py "$BLIF_FILE" "$HMETIS_PATH" "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo -e "❌ Error: blif2rent.py failed!"
    exit 1
fi
echo -e "✔️ Finished processing BLIF file.\n"

# Step 2: Generate Rent's Rule Visualization
echo "========== 📊 Running Rent's Rule Visualization =========="
python3 ./src/rent2viz.py "$RENT_JSON_FILE" "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    echo -e "❌ Error: rent2viz.py failed!"
    exit 1
fi

# Show final output locations
echo -e "\n========== ✅ Processing Completed Successfully! =========="
echo -e "📂 Results saved in:"
echo -e "\t🔹 Rent JSON File: $RENT_JSON_FILE"
echo -e "\t📊 Visualization Output: Saved in $OUTPUT_DIR\n"
