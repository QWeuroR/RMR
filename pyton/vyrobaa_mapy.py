import sys

filepath = "/home/qweuror/FEI/ING/2_S/RMR/pyton/raw_map.txt"
outpath = "/home/qweuror/FEI/ING/2_S/RMR/pyton/vyrobenamapa.txt"

print(f"Reading {filepath}...")
with open(filepath, "r") as f:
    lines = f.readlines()

print("Replacing values (0 -> X, 255 -> 0)...")
with open(outpath, "w") as f_out:
    for line in lines:
        row_vals = line.strip().split()
        if not row_vals: continue
        
        new_row = []
        for val_str in row_vals:
            if val_str == "0":
                new_row.append("X")
            else:
                new_row.append("0")  # Treat 255 (or -1/other expected free space values) as 0
                
        f_out.write(" ".join(new_row) + "\n")

print(f"Done! Saved processing to {outpath}")
