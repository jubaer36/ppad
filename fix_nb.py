import json

with open('ppad_kaggle.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            # Check for matplotlib import
            has_mpl = any('import matplotlib' in line for line in source)
            has_inline = any('%matplotlib inline' in line for line in source)
            if has_mpl and not has_inline:
                cell['source'] = ['%matplotlib inline\n'] + source
                source = cell['source'] # Update reference
                
            # Replace bug line
            for i, line in enumerate(source):
                if "save_path    = vis_dir / grid_fname," in line:
                    source[i] = line.replace("save_path    = vis_dir / grid_fname,", "save_path    = vis_dir / grid_fname if not config.no_vis else None,")

# Add Zip Cell
has_zip = any('shutil.make_archive' in "".join(c['source']) for c in nb.get('cells', []) if c['cell_type'] == 'code')
if not has_zip:
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Package Outputs for Download\n",
            "\n",
            "When running via **Save Version (Commit)** on Kaggle, downloading hundreds of image files individually can be tedious or hit file limits. This cell zips the `visualizations` and `checkpoints` folders so you can easily download them as single `.zip` files from the Kaggle Output panel."
        ]
    }
    zip_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Zip the output directories to make them easy to download from Kaggle Outputs\n",
            "import shutil\n",
            "import os\n",
            "\n",
            "if os.path.exists(Config.vis_dir):\n",
            "    print('Zipping visualizations...')\n",
            "    shutil.make_archive('visualizations_output', 'zip', Config.vis_dir)\n",
            "    print('Created visualizations_output.zip')\n",
            "\n",
            "if os.path.exists(Config.output_dir):\n",
            "    print('Zipping checkpoints and results...')\n",
            "    shutil.make_archive('checkpoints_output', 'zip', Config.output_dir)\n",
            "    print('Created checkpoints_output.zip')\n"
        ]
    }
    nb['cells'].extend([md_cell, zip_cell])

with open('ppad_kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook properly fixed!")
