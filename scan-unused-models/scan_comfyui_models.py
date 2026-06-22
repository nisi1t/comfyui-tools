"""
ComfyUI Model Auditor
=====================
Run this script from the ComfyUI portable root folder:

    python scan_comfyui_models.py

Author: Nisipeanu Mihai/
Github: https://github.com/nisi1t/comfyui-tools

Output JSON structure
---------------------
{
  "workflow_models": {
    "loras": [
      { "filename": "Qwen-image_edit.safetensors", "found": true,  "used_in": ["my_workflow.json"] },
      { "filename": "missing_lora.safetensors",    "found": false, "used_in": ["other.json"] }
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

MODEL_KEYS = {
    "ckpt_name", "unet_name", "model_name", "vae_name",
    "clip_name", "clip_name1", "clip_name2", "lora_name",
    "control_net_name", "style_model_name", "gligen_textbox_model",
    "upscale_model", "video_model", "audio_model",
    "encoder_name", "decoder_name", "motion_module",
    "ip_adapter_name", "ipadapter", "image_encoder", "embedding_name",
}

MODEL_EXTENSIONS = {
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin",
    ".gguf", ".onnx", ".sft",
}

# ---------------------------------------------------------------------------
# Disk index
# ---------------------------------------------------------------------------

def build_models_on_disk(models_dir: Path):
    """
    Returns:
      by_key      : { "loras/file.safetensors": Path }
      by_filename : { "file.safetensors": ["loras/file.safetensors", ...] }
    """
    by_key      = {}
    by_filename = defaultdict(list)

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
    """Return canonical key (folder/file) or None."""
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

# ---------------------------------------------------------------------------
# Extraction from one workflow
# ---------------------------------------------------------------------------

def extract_refs(workflow, by_key, by_filename):
    """
    Collect raw model filename strings from a workflow dict.
    Three strategies, in priority order:

      1. Top-level "models" array  (v1.0 schema, most explicit)
      2. widgets_values strings    (app/UI format — main source)
      3. Named "inputs" keys       (API/prompt format — fallback)
    """
    found = []
    seen  = set()

    def add(val):
        v = val.strip()
        if v and v not in seen:
            seen.add(v)
            found.append(v)

    # ── Strategy 1: top-level "models" array (newer ComfyUI versions) ──────
    for entry in workflow.get("models", []):
        if isinstance(entry, dict):
            name = entry.get("name", "")
            if name:
                _, ext = os.path.splitext(name)
                if ext.lower() in MODEL_EXTENSIONS:
                    add(name)

    # ── Strategy 2: widgets_values (app/UI format — your saved workflows) ──
    nodes_list = workflow.get("nodes", [])
    if not nodes_list and "workflow" in workflow:
        nodes_list = workflow["workflow"].get("nodes", [])

    for node in nodes_list:
        if not isinstance(node, dict):
            continue
        for item in node.get("widgets_values", []):
            if not isinstance(item, str) or not item.strip():
                continue
            item = item.strip()
            # Accept if it matches something on disk OR has a model extension
            if find_model_on_disk(item, by_key, by_filename):
                add(item)
            else:
                _, ext = os.path.splitext(item)
                if ext.lower() in MODEL_EXTENSIONS:
                    add(item)

    # ── Strategy 3: named inputs (API/prompt format) ────────────────────────
    if isinstance(workflow, dict):
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            for key, val in inputs.items():
                if key in MODEL_KEYS and isinstance(val, str):
                    _, ext = os.path.splitext(val)
                    if ext.lower() in MODEL_EXTENSIONS:
                        add(val)

    return found

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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

    # ── Step 1: index disk ───────────────────────────────────────────────────
    print("\nScanning models folder …")
    by_key, by_filename = build_models_on_disk(models_dir)
    print(f"  Found {len(by_key)} model file(s) on disk.")

    # ── Step 2: read workflows ───────────────────────────────────────────────
    print("\nReading workflows …")

    # usage_map: canonical_key -> set of workflow filenames
    usage_map = defaultdict(set)

    workflow_files = list(workflows_dir.rglob("*.json"))
    print(f"  Found {len(workflow_files)} workflow file(s).")

    for wf_path in workflow_files:
        wf_name = wf_path.name
        try:
            with wf_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"  [WARN] Could not parse {wf_name}: {exc}")
            continue

        refs = extract_refs(data, by_key, by_filename)
        print(f"  {wf_name}: {len(refs)} model reference(s) found")

        for raw in refs:
            matched = find_model_on_disk(raw, by_key, by_filename)
            if matched:
                usage_map[matched].add(wf_name)
            else:
                # Not on disk — store with a placeholder folder "unknown"
                # unless the raw value already contains a slash
                norm = raw.replace("\\", "/")
                usage_map[norm].add(wf_name)

    # ── Step 3: build workflow_models  { folder -> [ {filename, found, used_in} ] }
    # Group by folder
    folder_map = defaultdict(list)   # folder -> list of entry dicts

    for key, wf_set in usage_map.items():
        parts    = key.split("/")
        folder   = "/".join(parts[:-1]) if len(parts) >= 2 else "unknown"
        filename = parts[-1]
        folder_map[folder].append({
            "filename": filename,
            "found":    key in by_key,
            "used_in":  sorted(wf_set),
        })

    # Sort entries inside each folder alphabetically
    workflow_models = {
        folder: sorted(entries, key=lambda e: e["filename"].lower())
        for folder, entries in sorted(folder_map.items())
    }

    # ── Step 4: orphaned_models  { folder -> [filename, ...] }
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


if __name__ == "__main__":
    main()
