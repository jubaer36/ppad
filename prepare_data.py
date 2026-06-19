import os
from pathlib import Path

def prepare_mvtec():
    src_dir = Path("/home/rgb/Desktop/research/ad/FoundAD/assets/mvtec")
    dest_dir = Path("/home/rgb/Desktop/research/ad/ppad/data/mvtec")
    
    bottle_src = src_dir / "bottle"
    bottle_dest = dest_dir / "bottle"
    
    # Create directories
    (bottle_dest / "train" / "good").mkdir(parents=True, exist_ok=True)
    (bottle_dest / "test" / "good").mkdir(parents=True, exist_ok=True)
    (bottle_dest / "test" / "contamination").mkdir(parents=True, exist_ok=True)
    (bottle_dest / "test" / "broken_small").mkdir(parents=True, exist_ok=True)
    (bottle_dest / "test" / "broken_large").mkdir(parents=True, exist_ok=True)
    
    # Symlinks/copies
    symlink_or_copy(bottle_src / "good" / "000.png", bottle_dest / "train" / "good" / "000.png")
    symlink_or_copy(bottle_src / "good" / "000.png", bottle_dest / "test" / "good" / "000.png")
    symlink_or_copy(bottle_src / "contamination" / "000.png", bottle_dest / "test" / "contamination" / "000.png")
    symlink_or_copy(bottle_src / "broken_small" / "004.png", bottle_dest / "test" / "broken_small" / "004.png")
    symlink_or_copy(bottle_src / "broken_large" / "002.png", bottle_dest / "test" / "broken_large" / "002.png")
    print("MVTec prepared at", dest_dir)

def prepare_visa():
    src_dir = Path("/home/rgb/Desktop/research/ad/FoundAD/assets/visa")
    dest_dir = Path("/home/rgb/Desktop/research/ad/ppad/data/visa")
    
    chewinggum_src = src_dir / "chewinggum"
    chewinggum_dest = dest_dir / "chewinggum"
    
    # Create directories
    (chewinggum_dest / "Data" / "Images" / "Normal").mkdir(parents=True, exist_ok=True)
    (chewinggum_dest / "Data" / "Images" / "Anomaly").mkdir(parents=True, exist_ok=True)
    (chewinggum_dest / "Data" / "Masks" / "Anomaly").mkdir(parents=True, exist_ok=True)
    
    # Symlinks/copies
    symlink_or_copy(chewinggum_src / "ok" / "006.JPG", chewinggum_dest / "Data" / "Images" / "Normal" / "006.JPG")
    symlink_or_copy(chewinggum_src / "ko" / "000.JPG", chewinggum_dest / "Data" / "Images" / "Anomaly" / "000.JPG")
    print("VisA prepared at", dest_dir)

def symlink_or_copy(src, dest):
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.symlink(src, dest)
    except Exception:
        import shutil
        shutil.copy(src, dest)

if __name__ == "__main__":
    prepare_mvtec()
    prepare_visa()
