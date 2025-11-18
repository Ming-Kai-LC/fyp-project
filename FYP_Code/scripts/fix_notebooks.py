"""
Fix PyTorch 2.7.1 compatibility issues in notebooks
Removes 'verbose=True' from ReduceLROnPlateau (deprecated parameter)
"""

import json
from pathlib import Path

notebooks_dir = Path("notebooks")
notebooks_to_fix = [
    "04_baseline_test.ipynb",
    "04_baseline_test_FULL.ipynb",
]

for notebook_name in notebooks_to_fix:
    notebook_path = notebooks_dir / notebook_name

    if not notebook_path.exists():
        print(f"[SKIP] {notebook_name} not found")
        continue

    print(f"[FIX] {notebook_name}")

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Fix each cell
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            if 'verbose=True' in source and 'ReduceLROnPlateau' in source:
                # Remove verbose=True parameter
                new_source = source.replace(',\n    verbose=True', '')
                new_source = new_source.replace(', verbose=True', '')
                new_source = new_source.replace(',verbose=True', '')

                cell['source'] = new_source.split('\n')
                # Keep newlines at end of lines except last
                cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line
                                 for i, line in enumerate(cell['source'])]
                modified = True
                print(f"  - Fixed ReduceLROnPlateau verbose parameter")

    if modified:
        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  [SAVED] {notebook_name}")
    else:
        print(f"  [SKIP] No changes needed")

print("\n[DONE] All notebooks fixed!")
