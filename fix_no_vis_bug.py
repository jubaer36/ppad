import json

with open('ppad_kaggle.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "save_path    = vis_dir / grid_fname," in line:
                source[i] = "                    save_path    = save_path.parent / grid_fname if save_path is not None else None,\n"
        cell['source'] = source

with open('ppad_kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
