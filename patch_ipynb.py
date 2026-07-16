import json

with open('ppad_kaggle.ipynb', 'r') as f:
    nb = json.load(f)

found = False
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('import matplotlib' in line for line in source):
            # Check if %matplotlib inline is already there
            if not any('%matplotlib inline' in line for line in source):
                cell['source'] = ['%matplotlib inline\n'] + source
            found = True
            break

with open('ppad_kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Patched:", found)
