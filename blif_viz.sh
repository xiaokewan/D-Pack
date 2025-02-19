#!/bin/bash

# Exit immediately on error
set -e

# Usage function
usage() {
    echo "Usage: $0 <blif_file_or_directory>"
    echo "If a directory is provided, all .blif files inside it will be processed."
    exit 1
}

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    usage
fi

# Define input
INPUT=$1

# Define paths
HMETIS_PATH="./hmetis/hmetis"

# Function to process a single BLIF file
process_blif_file() {
    local BLIF_FILE=$1
    local BASE_NAME=$(basename "$BLIF_FILE" .blif)
    local RELATIVE_DIR=$(dirname "$BLIF_FILE")
    local OUTPUT_DIR="./results/${RELATIVE_DIR}/${BASE_NAME}/"
    local RENT_JSON_FILE="${OUTPUT_DIR}${BASE_NAME}.blif.json"

    mkdir -p "$OUTPUT_DIR"

    echo -e  "\nProcessing BLIF File: $BLIF_FILE"
    echo "Output Directory: $OUTPUT_DIR"

    # Step 1: Convert BLIF to Rent's Rule JSON
    python3 ./src/blif2rent.py "$BLIF_FILE" "$HMETIS_PATH" "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: blif2rent.py failed for $BLIF_FILE"
        exit 1
    fi

    # Step 2: Generate Rent's Rule Visualization
    python3 ./src/rent2viz.py "$RENT_JSON_FILE" "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: rent2viz.py failed for $BLIF_FILE"
        exit 1
    fi

    echo "Completed: $BLIF_FILE"
    echo -e "Results saved in: $OUTPUT_DIR\n"
}

# If the input is a file, process it
if [ -f "$INPUT" ]; then
    if [[ "$INPUT" == *.blif ]]; then
        process_blif_file "$INPUT"
    else
        echo "Error: The provided file is not a .blif file"
        exit 1
    fi
elif [ -d "$INPUT" ]; then
    # If the input is a directory, find all .blif files and process them
    find "$INPUT" -type f -name "*.blif" | while read -r blif_file; do
        process_blif_file "$blif_file"
    done
else
    echo "Error: Invalid input. Please provide a .blif file or a directory."
    exit 1
fi


