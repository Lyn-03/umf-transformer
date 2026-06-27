import json
from pathlib import Path

def patch_packages():
    notebook_path = Path("ufm-training.ipynb")
    if not notebook_path.exists():
        notebook_path = Path("src/ufm-training.ipynb")

    if not notebook_path.exists():
        print("Error: Could not find ufm-training.ipynb in current directory or src/")
        return

    print(f"Reading {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    patched = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            for i, line in enumerate(source):
                if "!apt-get update && apt-get install -y" in line:
                    source[i] = "!apt-get update && apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended cm-super texlive-bibtex-extra texlive-publishers texlive-lang-french texlive-science\n"
                    patched = True
                    print("Patched apt-get line in notebook.")
                    break

    if patched:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("Notebook updated successfully.")
    else:
        print("Error: Could not find cell containing apt-get line.")

if __name__ == "__main__":
    patch_packages()
