#!/usr/bin/env python3
"""
Split data/combined.json into individual chunk JSON files.

Each output file is named <chunk_id>.json and is written to data/chunks/.
Only chunks that have a 'title' field (i.e. annotated ManualChunk objects)
are exported; plain TextItem objects without a title are skipped.
"""

import json
import os
import sys


def split_chunks(input_path: str, output_dir: str) -> int:
    """Read *input_path* and write one JSON file per chunk into *output_dir*.

    Returns the number of files written.
    """
    with open(input_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    chunks = data.get("chunks", [])
    if not chunks:
        print("No chunks found in input file.", file=sys.stderr)
        return 0

    os.makedirs(output_dir, exist_ok=True)

    written = 0
    for chunk in chunks:
        # Skip plain TextItem objects that have no title
        if "title" not in chunk:
            continue

        chunk_id = chunk.get("id")
        if not chunk_id:
            print(f"Skipping chunk without 'id': {chunk.get('title')}", file=sys.stderr)
            continue
        out_path = os.path.join(output_dir, f"{chunk_id}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(chunk, fh, ensure_ascii=False, indent=2)
        written += 1

    return written


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "combined.json")
    output_directory = os.path.join(base_dir, "chunks")

    count = split_chunks(input_file, output_directory)
    print(f"Written {count} chunk file(s) to '{output_directory}'.")
