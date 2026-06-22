"""
ComfyUI Model Auditor
=====================
Run this script from the ComfyUI portable root folder:

    python scan_comfyui_models.py

Author: Nisipeanu Mihai/
Github: https://github.com/nisi1t/comfyui-tools

Add --debug to see per-node detail for every workflow:

    python scan_comfyui_models.py --debug

Output JSON structure
---------------------
{
  "workflow_models": {
    "loras": [
      { "filename": "Qwen-image_edit.safetensors", "found": true,  "used_in": ["subfolder/my_workflow.json"] }
    ],
    "diffusion_models": [
      { "filename": "flux1-fill-dev.safetensors",  "found": true,  "used_in": ["flux.json"] }
    ]
  },
  "orphaned_models": {
    "checkpoints": ["old_test.ckpt", "unused.safetensors"],
    "loras":       ["forgotten_lora.safetensors"]
  }
}
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKFLOWS_DIR = Path("ComfyUI") / "user" / "default" / "workflows"
MODELS_DIR    = Path("ComfyUI") / "models"
OUTPUT_FILE   = Path("model_audit.json")

MODEL_EXTENSIONS = {
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin",
    ".gguf", ".onnx", ".sft",
}

# ---------------------------------------------------------------------------
# Disk index
# ---------------------------------------------------------------------------

def build_models_on_disk(models_dir: Path):
    by_key      = {}                    # "loras/file.safetensors" -> Path
    by_filename = defaultdict(list)     # "file.safetensors"       -> ["loras/file.safetensors"]

    if not models_dir.is_dir():
        return by_key, by_filename

    for root, _dirs, files in os.walk(models_dir):
        root_path = Path(root)
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext.lower() not in MODEL_EXTENSIONS:
                continue
            rel = root_path.relative_to(models_dir)
            key = (rel / fname).as_posix()
            by_key[key] = root_path / fname
            by_filename[fname].append(key)

    return by_key, by_filename


def find_model_on_disk(raw, by_key, by_filename):
    """Return canonical 'folder/filename' key or None."""
    norm = raw.replace("\\", "/").strip()
    if norm in by_key:
        return norm
    basename = Path(norm).name
    candidates = by_filename.get(basename, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        folder_hint = Path(norm).parent.as_posix()
        if folder_hint and folder_hint != ".":
            for c in candidates:
                if c.startswith(folder_hint + "/"):
                    return c
        return candidates[0]
    return None


def iter_strings(obj):
    """Yield every string value in a loaded workflow JSON object."""
    stack = [obj]
    seen = set()

    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)

            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, str):
            yield current


def build_model_token_map(by_key):
    token_map = defaultdict(set)

    for disk_key in by_key:
        parts = disk_key.split("/")
        filename = parts[-1]
        token_map[disk_key].add(disk_key)
        token_map[filename].add(disk_key)

        if len(parts) > 2:
            token_map["/".join(parts[1:])].add(disk_key)

    return token_map


# ---------------------------------------------------------------------------
# Extract model refs from one workflow JSON
# ---------------------------------------------------------------------------

def extract_refs(workflow, by_key, by_filename, debug=False, wf_label=""):
    """
    Search every string in the workflow for models that exist on disk.

    This is intentionally node-type agnostic: if ComfyUI or a custom node stores
    a model reference under a new key, the model is still counted as used when
    its known filename/path appears anywhere in the workflow JSON.
    """
    found = []
    seen  = set()
    token_map = build_model_token_map(by_key)

    def add(disk_key, source):
        if disk_key not in seen:
            seen.add(disk_key)
            found.append(disk_key)
            if debug:
                print(f"      [+] {source}: {disk_key!r}")

    workflow_strings = list(iter_strings(workflow))
    if debug:
        print(f"    string values scanned: {len(workflow_strings)}")

    for value in workflow_strings:
        norm = value.replace("\\", "/").strip()
        if not norm:
            continue

        for disk_key in token_map.get(norm, []):
            add(disk_key, "exact-string")

        basename = Path(norm).name
        if basename != norm:
            for disk_key in token_map.get(basename, []):
                if norm == disk_key or norm.endswith("/" + disk_key):
                    add(disk_key, "path-suffix")

    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    debug = "--debug" in sys.argv

    root          = Path.cwd()
    workflows_dir = root / WORKFLOWS_DIR
    models_dir    = root / MODELS_DIR
    output_file   = root / OUTPUT_FILE

    if not workflows_dir.is_dir():
        print(f"[ERROR] Workflows folder not found:\n        {workflows_dir}")
        print("        Run this script from the ComfyUI portable root folder.")
        sys.exit(1)

    if not models_dir.is_dir():
        print(f"[ERROR] Models folder not found:\n        {models_dir}")
        sys.exit(1)

    print(f"Workflows : {workflows_dir}")
    print(f"Models    : {models_dir}")
    if debug:
        print("  [debug mode on]")

    # ── Step 1: index disk ───────────────────────────────────────────────────
    print("\nScanning models folder …")
    by_key, by_filename = build_models_on_disk(models_dir)
    print(f"  Found {len(by_key)} model file(s) on disk.")

    # ── Step 2: read workflows ───────────────────────────────────────────────
    print("\nReading workflows …")

    # usage_map: canonical_key -> set of workflow relative paths
    usage_map = defaultdict(set)

    workflow_files = list(workflows_dir.rglob("*.json"))
    print(f"  Found {len(workflow_files)} workflow file(s).")

    for wf_path in workflow_files:
        # Store path relative to workflows_dir so subfolder is preserved,
        # e.g. "subfolder/my_workflow.json" instead of just "my_workflow.json"
        wf_label = wf_path.relative_to(workflows_dir).as_posix()

        try:
            with wf_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"  [WARN] Could not parse {wf_label}: {exc}")
            continue

        if debug:
            print(f"\n  --- {wf_label} ---")

        refs = extract_refs(data, by_key, by_filename, debug=debug, wf_label=wf_label)
        print(f"  {wf_label}: {len(refs)} model reference(s) found")

        for raw in refs:
            matched = find_model_on_disk(raw, by_key, by_filename)
            if matched:
                usage_map[matched].add(wf_label)
            else:
                norm = raw.replace("\\", "/")
                usage_map[norm].add(wf_label)

    # ── Step 3: build workflow_models grouped by folder ──────────────────────
    folder_map = defaultdict(list)

    for key, wf_set in usage_map.items():
        parts    = key.split("/")
        folder   = "/".join(parts[:-1]) if len(parts) >= 2 else "unknown"
        filename = parts[-1]
        folder_map[folder].append({
            "filename": filename,
            "found":    key in by_key,
            "used_in":  sorted(wf_set),
        })

    workflow_models = {
        folder: sorted(entries, key=lambda e: e["filename"].lower())
        for folder, entries in sorted(folder_map.items())
    }

    # ── Step 4: orphaned_models grouped by folder ────────────────────────────
    used_keys = set(usage_map.keys())

    orphan_folder_map = defaultdict(list)
    for disk_key in by_key:
        if disk_key not in used_keys:
            parts  = disk_key.split("/")
            folder = "/".join(parts[:-1]) if len(parts) >= 2 else "unknown"
            orphan_folder_map[folder].append(parts[-1])

    orphaned_models = {
        folder: sorted(filenames, key=str.lower)
        for folder, filenames in sorted(orphan_folder_map.items())
    }

    # ── Step 5: write report ─────────────────────────────────────────────────
    report = {
        "workflow_models": workflow_models,
        "orphaned_models": orphaned_models,
    }
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_wf  = sum(len(v) for v in workflow_models.values())
    found_wf  = sum(1 for v in workflow_models.values() for e in v if e["found"])
    total_orp = sum(len(v) for v in orphaned_models.values())

    print(f"\n{'='*55}")
    print(f"  Workflow models referenced : {total_wf}")
    print(f"    ✔ found on disk          : {found_wf}")
    print(f"    ✘ missing on disk        : {total_wf - found_wf}")
    print(f"  Orphaned model files       : {total_orp}")
    print(f"{'='*55}")
    print(f"\nReport written to: {output_file}")
    if total_orp > 0 and not debug:
        print("Tip: run with --debug to see exactly what each workflow yields.")


if __name__ == "__main__":
    main()
