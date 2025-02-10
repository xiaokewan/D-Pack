#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <blif_file>"
    exit 1
fi

BLIF_FILE=$1
HMETIS_PATH="./hmetis/hmetis"
echo "BLIF_FILE: $BLIF_FILE"

BASE_NAME=$(basename "$BLIF_FILE" .blif)
echo "BASE_NAME: $BASE_NAME"

OUTPUT_DIR="./results/${BASE_NAME}/"
echo "OUTPUT_DIR: $OUTPUT_DIR"

RENT_JSON_FILE="${OUTPUT_DIR}${BASE_NAME}.blif.json"

echo "RENT_JSON_FILE: $RENT_JSON_FILE"

mkdir -p "$OUTPUT_DIR"

echo "Processing $BLIF_FILE with netlist2rent.py..."
python3 ./src/blif2rent.py "$BLIF_FILE" "$HMETIS_PATH"


echo "Generating visualization for $RENT_JSON_FILE..."
python3 ./src/rent2viz.py "$RENT_JSON_FILE" "$OUTPUT_DIR"

echo "Processing completed. Results are saved in $OUTPUT_DIR"
