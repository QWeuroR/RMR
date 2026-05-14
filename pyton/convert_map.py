import re
import os

filepath = "/home/qweuror/FEI/ING/2_S/RMR/pyton/raw_map.yaml"
outpath = "/home/qweuror/FEI/ING/2_S/RMR/pyton/raw_map.txt"

print(f"Reading {filepath}...")
with open(filepath, "r") as f:
    content = f.read()

print("Extracting data...")
match = re.search(r'data:\s*\[(.*?)\]', content, re.DOTALL)
if match:
    data_str = match.group(1)
    numbers = [x.strip() for x in data_str.split(',') if x.strip()]
    print(f"Found {len(numbers)} numbers.")
    
    rows = 1362
    cols = 1204
    
    if len(numbers) >= rows * cols:
        print(f"Writing to {outpath}...")
        with open(outpath, "w") as f_out:
            for r in range(rows):
                start = r * cols
                end = start + cols
                row_data = numbers[start:end]
                f_out.write(" ".join(row_data) + "\n")
        print("Done!")
    else:
        print(f"Not enough data! Expected {rows * cols}, got {len(numbers)}")
else:
    print("Could not find data array in YAML file.")
