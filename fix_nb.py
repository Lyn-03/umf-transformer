"""
fix_nb.py — Patch the evaluation cell in ufm-training.ipynb.

Replaces the USE_SEPARATE placeholder branch in Step 5 (evaluation cell)
with a real call to evaluate_separate.py.

Run from the project root:
    python3 fix_nb.py

Or from the src/ directory:
    python3 ../fix_nb.py
"""
import json
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the notebook
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
CANDIDATES = [
    _HERE / "src" / "ufm-training.ipynb",
    _HERE / "ufm-training.ipynb",
]
NOTEBOOK = next((p for p in CANDIDATES if p.exists()), None)
if NOTEBOOK is None:
    raise FileNotFoundError(
        "Could not find ufm-training.ipynb. "
        "Run this script from the project root or src/ directory."
    )

BACKUP = NOTEBOOK.with_suffix(".ipynb.bak")

# ---------------------------------------------------------------------------
# The new source lines for the evaluation cell (Step 5 / "Run evaluation")
# ---------------------------------------------------------------------------
NEW_SOURCE = [
    "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "# Resolve best checkpoint: prefer phase2, fall back to phase1\n",
    "_ckpt_candidates = [\n",
    "    OUTPUT_DIR / 'best_model_phase2.pt',\n",
    "    OUTPUT_DIR / 'best_model_phase1.pt',\n",
    "    OUTPUT_DIR / 'shutdown_checkpoint_phase2.pt',\n",
    "    OUTPUT_DIR / 'shutdown_checkpoint_phase1.pt',\n",
    "]\n",
    "best_checkpoint = next((p for p in _ckpt_candidates if p.exists()), None)\n",
    "if best_checkpoint is None:\n",
    "    # last resort: any .pt in OUTPUT_DIR\n",
    "    _pts = sorted(OUTPUT_DIR.glob('*.pt'))\n",
    "    best_checkpoint = _pts[-1] if _pts else None\n",
    "if best_checkpoint is None:\n",
    "    raise FileNotFoundError(f'No checkpoint found in {OUTPUT_DIR}. Run training first.')\n",
    "print(f'Using checkpoint: {best_checkpoint}')\n",
    "\n",
    "if USE_SEPARATE:\n",
    "    # --- Automated unimodal evaluation (separate-dataset mode) ---\n",
    "    # Evaluates face encoder on CASIA-WebFace and fingerprint encoder on SOCOFing\n",
    "    # independently, producing figures/ and tables/ for LaTeX.\n",
    "    print('Separate dataset mode: running unimodal evaluation via evaluate_separate.py')\n",
    "    eval_sep_cmd = [\n",
    "        sys.executable, str(SRC_DIR / 'evaluate_separate.py'),\n",
    "        '--face_path', str(FACE_PATH),\n",
    "        '--fingerprint_path', str(FINGERPRINT_PATH),\n",
    "        '--model_path', str(best_checkpoint),\n",
    "        '--output_dir', str(RESULTS_DIR),\n",
    "        '--embed_dim', str(EMBED_DIM),\n",
    "        '--image_size', str(IMAGE_SIZE),\n",
    "        '--batch_size', str(BATCH_SIZE),\n",
    "        '--num_workers', str(NUM_WORKERS),\n",
    "        '--device', DEVICE,\n",
    "    ]\n",
    "    run(eval_sep_cmd, cwd=str(SRC_DIR))\n",
    "else:\n",
    "    MODEL_PATH = OUTPUT_DIR / 'checkpoints' / 'best_model.pth'\n",
    "    if not MODEL_PATH.exists():\n",
    "        raise FileNotFoundError(f'Expected best model not found: {MODEL_PATH}')\n",
    "\n",
    "    # Use evaluate.py directly — it produces figures/ and tables/ with PDFs + .tex\n",
    "    eval_cmd = [\n",
    "        sys.executable, 'evaluate.py',\n",
    "        '--model_path', str(MODEL_PATH),\n",
    "        '--dataset_path', str(DATASET_PATH),\n",
    "        '--output_dir', str(RESULTS_DIR),\n",
    "        '--batch_size', str(BATCH_SIZE),\n",
    "        '--image_size', str(IMAGE_SIZE),\n",
    "        '--num_workers', str(NUM_WORKERS),\n",
    "        '--device', DEVICE,\n",
    "    ]\n",
    "    run(eval_cmd, cwd=str(SRC_DIR))",
]

# ---------------------------------------------------------------------------
# Fingerprint of the target cell (must contain all of these strings)
# ---------------------------------------------------------------------------
MARKER_LINES = [
    "RESULTS_DIR.mkdir(parents=True, exist_ok=True)",
    "MODEL_PATH = OUTPUT_DIR / 'checkpoints' / 'best_model.pth'",
    "Separate dataset mode: no shared test set for full evaluation.",
]


def cell_matches(source_list: list) -> bool:
    joined = "".join(source_list)
    return all(m in joined for m in MARKER_LINES)


def main() -> None:
    print(f"Notebook : {NOTEBOOK}")

    # Backup
    shutil.copy2(NOTEBOOK, BACKUP)
    print(f"Backup   : {BACKUP}")

    with open(NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    patched = False
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if cell_matches(src):
            print(f"  Found target cell at index {i} — patching ...")
            cell["source"] = NEW_SOURCE
            cell["outputs"] = []
            cell["execution_count"] = None
            patched = True
            break

    if not patched:
        print("ERROR: Could not find the target evaluation cell. No changes made.")
        print("Expected cell to contain all of:")
        for m in MARKER_LINES:
            print(f"  - {repr(m)}")
        return

    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print("Notebook patched successfully.")
    print("Step 5 now calls evaluate_separate.py when USE_SEPARATE=True.")


if __name__ == "__main__":
    main()
