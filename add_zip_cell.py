import json

with open('ppad_kaggle.ipynb', 'r') as f:
    nb = json.load(f)

# Check if zip cell already exists
has_zip = False
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        if 'shutil.make_archive' in "".join(cell['source']):
            has_zip = True
            break

if not has_zip:
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
    
    # Add a markdown header for it
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Package Outputs for Download\n",
            "\n",
            "When running via **Save Version (Commit)** on Kaggle, downloading hundreds of image files individually can be tedious or hit file limits. This cell zips the `visualizations` and `checkpoints` folders so you can easily download them as single `.zip` files from the Kaggle Output panel."
        ]
    }
    
    nb['cells'].append(md_cell)
    nb['cells'].append(zip_cell)
    
    with open('ppad_kaggle.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    print("Zip cells added.")
else:
    print("Zip cell already exists.")
