#!/usr/bin/env python3
"""Safely fix grokking detection calls in Nanda notebook."""

import json

# Load notebook
with open('analyze_nanda_experiments.ipynb', 'r') as f:
    nb = json.load(f)

count = 0

# Iterate through code cells
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source_lines = cell['source']
    i = 0
    
    while i < len(source_lines):
        line = source_lines[i]
        
        # Look for the old pattern
        if 'grok_epoch_idx = detect_grokking_epoch(train_acc, test_acc)' in line:
            # Check if next line has the mapping
            if i + 1 < len(source_lines):
                next_line = source_lines[i + 1]
                
                # Pattern 1: Standard mapping on next line
                if 'grok_epoch = epochs[grok_epoch_idx] if grok_epoch_idx is not None else None' in next_line:
                    # Get indentation from current line
                    indent = len(line) - len(line.lstrip())
                    # Replace both lines with single new line
                    new_line = ' ' * indent + 'grok_epoch = detect_grokking_epoch(train_acc, test_acc, epochs=epochs)\n'
                    source_lines[i] = new_line
                    del source_lines[i + 1]
                    count += 1
                    print(f"Fixed pattern with {indent} spaces indentation")
                    continue
                
                # Pattern 2: epochs defined on next line, mapping on line after
                elif 'epochs = np.array(history.get(' in next_line:
                    if i + 2 < len(source_lines):
                        third_line = source_lines[i + 2]
                        if 'grok_epoch = epochs[grok_epoch_idx]' in third_line:
                            # Keep epochs line, update the other two
                            indent = len(line) - len(line.lstrip())
                            source_lines[i + 1] = source_lines[i + 1]  # Keep epochs line as is
                            new_line = ' ' * indent + 'grok_epoch = detect_grokking_epoch(train_acc, test_acc, epochs=epochs)\n'
                            # Delete old detection line and old mapping line
                            del source_lines[i]  # Remove old grok_epoch_idx line
                            del source_lines[i + 1]  # Remove old grok_epoch mapping line (now at i+1)
                            count += 1
                            print(f"Fixed reversed pattern with {indent} spaces indentation")
                            continue
        
        i += 1

print(f"\nTotal fixes: {count}")

# Save
with open('analyze_nanda_experiments.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully!")

