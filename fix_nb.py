"""Patch ufm-training.ipynb: set NUM_WORKERS=4."""
import json
from pathlib import Path

nb_path = Path(__file__).parent / "src" / "ufm-training.ipynb"
nb = json.loads(nb_path.read_text())

changed = 0
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        new_src = []
        for line in cell["source"]:
            if "NUM_WORKERS = 2" in line:
                line = line.replace("NUM_WORKERS = 2", "NUM_WORKERS = 4")
                changed += 1
            new_src.append(line)
        cell["source"] = new_src

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Done — replaced {changed} occurrence(s).")
