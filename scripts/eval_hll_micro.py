import json
import subprocess
import h5py
import numpy as np
import h3

def main():
    features_h5 = "datasets/1501677363692556/clip_features.h5"
    group = "clip"
    resolution = 10
    precision = 10
    
    # 1. Get true counts
    with h5py.File(features_h5, 'r') as f:
        lats = f[f'{group}/lat'][:]
        lngs = f[f'{group}/lng'][:]
        
    true_counts = {}
    for lat, lng in zip(lats, lngs):
        cell = h3.latlng_to_cell(lat, lng, resolution)
        true_counts[cell] = true_counts.get(cell, 0) + 1
        
    # 2. Get HLL estimates from psm binary
    cmd = [
        "targets/psm", features_h5, group, "1000000", str(resolution), "1", str(precision), "-j"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = json.loads(proc.stdout)
    
    # Sort by true count descending to show the most active cells first
    tiles = sorted(out["tiles"], key=lambda t: true_counts.get(t["cell"], 0), reverse=True)
    
    print("| Cell | True count | HLL estimate (p=10) | Relative error |")
    print("|---|---|---|---|")
    
    errors = []
    for tile in tiles:
        cell = tile["cell"]
        if cell not in true_counts:
            continue
        true_c = true_counts[cell]
        hll_c = tile["total"]
        
        rel_error = abs(hll_c - true_c) / true_c
        errors.append(rel_error)
        print(f"| `{cell}` | {true_c} | {hll_c:.0f} | {rel_error:.1%} |")
        
    print(f"| **Median** | \u2014 | \u2014 | **{np.median(errors):.1%}** |")

if __name__ == '__main__':
    main()
