"""
ComfyUI Model Auditor
=====================
Run this script from the ComfyUI portable root folder:

    python scan_comfyui_models.py

Author: Nisipeanu Mihai/
Github: https://github.com/nisi1t/comfyui-tools

It will:
  1. Read every workflow JSON in  ComfyUI/user/default/workflows/
  2. Collect every model file referenced in those workflows
  3. Check which of those models actually exist under  ComfyUI/models/
  4. Flag models that exist on disk but are NOT used by any workflow (orphans)
  5. Write the full report to  model_audit.json  in the current directory

Output JSON structure
---------------------
{
  "workflow_models": {
    "diffusion_models/flux1-fill-dev.safetensors": {
      "folder":   "diffusion_models",
      "filename": "flux1-fill-dev.safetensors",
      "found":    true,
      "used_in":  ["txt2img_flux.json", ...]
    },
    ...
  },
  "orphaned_models": {
    "loras/old_test_lora.safetensors": {
      "folder":   "loras",
      "filename": "old_test_lora.safetensors"
    },
    ...
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

# Relative to the ComfyUI portable root (where this script lives)
WORKFLOWS_DIR = Path("ComfyUI") / "user" / "default" / "workflows"
MODELS_DIR    = Path("ComfyUI") / "models"
OUTPUT_FILE   = Path("model_audit.json")

# Node input keys that typically hold model filenames
# ComfyUI stores model names as plain filenames (sometimes with sub-paths)
MODEL_KEYS = {
    "ckpt_name",
    "unet_name",
    "model_name",
    "vae_name",
    "clip_name",
    "clip_name1",
    "clip_name2",
    "lora_name",
    "control_net_name",
    "style_model_name",
    "gligen_textbox_model",
    "upscale_model",
    "video_model",
    "audio_model",
    "encoder_name",
    "decoder_name",
    "sampler_name",     # usually a string enum, but include just in case
    "motion_module",
    "ip_adapter_name",
    "ipadapter",
    "image_encoder",
    "embedding_name",
    "filename_prefix",  # sometimes contains a model reference
}

# File extensions we consider "model files"
MODEL_EXTENSIONS = {
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin",
    ".gguf", ".onnx", ".sft",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_all_values(obj):
    """Recursively yield every value in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from iter_all_values(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_all_values(item)  # type: ignore[arg-type]


def extract_model_refs_from_workflow(workflow: dict) -> list[str]:
    """
    Return a list of raw model filename strings found in a workflow JSON.

    ComfyUI workflows are stored in two formats:
      • API format  – nodes are a dict keyed by node id, each having "inputs"
      • App format  – has a top-level "nodes" list with "widgets_values"

    We handle both by scanning all key/value pairs recursively and picking up
    any value whose key is in MODEL_KEYS and whose value looks like a filename
    with a model extension.
    """
    refs: list[str] = []

    def _check(key, value):
        if not isinstance(value, str):
            return
        if key in MODEL_KEYS:
            _, ext = os.path.splitext(value)
            if ext.lower() in MODEL_EXTENSIONS:
                refs.append(value)

    # API-format: flat dict of node objects
    if isinstance(workflow, dict):
        for _k, node in workflow.items():
            if isinstance(node, dict) and "inputs" in node:
                for inp_key, inp_val in node["inputs"].items():
                    _check(inp_key, inp_val)

    # App-format: "nodes" list
    if isinstance(workflow, dict) and "nodes" in workflow:
        for node in workflow.get("nodes", []):
            if not isinstance(node, dict):
                continue
            # widgets_values is a plain list; we can't know which is a model
            # name without correlating to node type, so we also walk properties
            props = node.get("properties", {})
            for pk, pv in props.items():
                _check(pk, pv)
            # Walk the full node dict for any named inputs
            for _k, v in iter_all_values(node):
                if isinstance(_k, str):
                    _check(_k, v)

    return refs


def build_models_on_disk(models_dir: Path) -> dict[str, Path]:
    """
    Return a mapping:   relative_key -> absolute_path
    where relative_key = "subfolder/filename.ext"  (using forward slashes)
    """
    result: dict[str, Path] = {}
    if not models_dir.is_dir():
        return result
    for root, _dirs, files in os.walk(models_dir):
        root_path = Path(root)
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext.lower() in MODEL_EXTENSIONS:
                rel = root_path.relative_to(models_dir)
                key = (rel / fname).as_posix()  # e.g. "loras/my_lora.safetensors"
                result[key] = root_path / fname
    return result


def find_model_on_disk(
    raw_name: str,
    disk_index: dict[str, Path],
) -> str | None:
    """
    Try to match a raw workflow model name against the disk index.

    ComfyUI stores names like:
      • "flux1-fill-dev.safetensors"           (just filename)
      • "diffusion_models/flux1-fill-dev.safetensors"  (with subfolder)

    Returns the matched key (e.g. "diffusion_models/flux1.safetensors") or None.
    """
    # Normalise separators
    normalised = raw_name.replace("\\", "/")

    # Exact match first
    if normalised in disk_index:
        return normalised

    # Match by filename only (no sub-path given in workflow)
    basename = Path(normalised).name
    candidates = [k for k in disk_index if Path(k).name == basename]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Prefer the one whose folder suffix matches the raw_name folder hint
        folder_hint = Path(normalised).parent.as_posix()
        if folder_hint and folder_hint != ".":
            for c in candidates:
                if c.startswith(folder_hint + "/"):
                    return c
        # Return the first match and note ambiguity
        return candidates[0]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = Path.cwd()
    workflows_dir = root / WORKFLOWS_DIR
    models_dir    = root / MODELS_DIR
    output_file   = root / OUTPUT_FILE

    # ---- Validate paths ----------------------------------------------------
    if not workflows_dir.is_dir():
        print(f"[ERROR] Workflows folder not found: {workflows_dir}")
        print("        Make sure you run this script from the ComfyUI portable root.")
        sys.exit(1)

    if not models_dir.is_dir():
        print(f"[ERROR] Models folder not found: {models_dir}")
        sys.exit(1)

    print(f"Workflows : {workflows_dir}")
    print(f"Models    : {models_dir}")

    # ---- Step 1: index all model files on disk -----------------------------
    print("\nScanning models folder …")
    disk_index = build_models_on_disk(models_dir)
    print(f"  Found {len(disk_index)} model file(s) on disk.")

    # ---- Step 2: read all workflow JSONs -----------------------------------
    print("\nReading workflows …")
    # model_key -> { folder, filename, found, used_in }
    workflow_models: dict[str, dict] = {}
    # model_key -> set of workflow filenames that use it
    usage_map: dict[str, set[str]] = defaultdict(set)

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

        refs = extract_model_refs_from_workflow(data)
        for raw_ref in refs:
            matched_key = find_model_on_disk(raw_ref, disk_index)

            if matched_key:
                key = matched_key
            else:
                # Use the raw name as the key; we'll mark it not found
                key = raw_ref.replace("\\", "/")

            usage_map[key].add(wf_name)

    # ---- Step 3: build workflow_models section -----------------------------
    for key, workflows in usage_map.items():
        parts = key.split("/")
        if len(parts) >= 2:
            folder   = "/".join(parts[:-1])
            filename = parts[-1]
        else:
            folder   = ""
            filename = parts[0]

        found = key in disk_index

        workflow_models[key] = {
            "folder":   folder,
            "filename": filename,
            "found":    found,
            "used_in":  sorted(workflows),
        }

    # ---- Step 4: orphaned models -------------------------------------------
    used_keys = set(workflow_models.keys())
    orphaned_models: dict[str, dict] = {}

    for disk_key in disk_index:
        if disk_key not in used_keys:
            parts = disk_key.split("/")
            if len(parts) >= 2:
                folder   = "/".join(parts[:-1])
                filename = parts[-1]
            else:
                folder   = ""
                filename = parts[0]

            orphaned_models[disk_key] = {
                "folder":   folder,
                "filename": filename,
            }

    # ---- Step 5: write JSON ------------------------------------------------
    report = {
        "workflow_models":  workflow_models,
        "orphaned_models":  orphaned_models,
    }

    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # ---- Summary -----------------------------------------------------------
    found_count   = sum(1 for m in workflow_models.values() if m["found"])
    missing_count = len(workflow_models) - found_count

    print(f"\n{'='*55}")
    print(f"  Workflow models referenced : {len(workflow_models)}")
    print(f"    ✔ found on disk          : {found_count}")
    print(f"    ✘ missing on disk        : {missing_count}")
    print(f"  Orphaned model files       : {len(orphaned_models)}")
    print(f"{'='*55}")
    print(f"\nReport written to: {output_file}")


if __name__ == "__main__":
    main()
