"""Build N rank-R LoRA adapters with UNTRAINED (seeded random) weights for capacity testing.

These are NOT trained adapters. They reproduce an adapter *shape* (rank, target
layers) so memory, residency, switching and latency can be measured. No quality claim.

Usage: make_adapters.py <model_dir> <out_dir> <count> <rank> <alpha> <suffix,suffix,...> [std] [name-prefix]
"""
import glob
import hashlib
import json
import os
import struct
import sys

import torch
from safetensors.torch import save_file

model_dir, out_dir, count, rank, alpha = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
suffixes = sys.argv[6].split(",")
std = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-3
prefix = sys.argv[8] if len(sys.argv) > 8 else "cap-r%d" % rank

shapes = {}
for path in sorted(glob.glob(os.path.join(model_dir, "model-*-of-*.safetensors"))):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    for key, meta in header.items():
        if key == "__metadata__" or not key.endswith(".weight") or len(meta["shape"]) != 2:
            continue
        if not key.startswith("language_model.backbone.layers."):
            continue
        module = key[: -len(".weight")]
        leaf = module.split(".")[-1]
        if ".experts." in module:  # routed experts excluded (512 per layer)
            continue
        if leaf in suffixes:
            shapes[module] = meta["shape"]  # [out, in]

params = sum(rank * (o + i) for o, i in shapes.values())
print("target modules: %d, params per adapter: %d (%.1f M)" % (len(shapes), params, params / 1e6))
summary = {"rank": rank, "alpha": alpha, "target_suffixes": suffixes, "target_module_count": len(shapes),
           "params_per_adapter": params, "weights": "untrained seeded random (std %g), bf16" % std, "adapters": []}
for idx in range(1, count + 1):
    name = "%s-%02d" % (prefix, idx)
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    gen = torch.Generator().manual_seed(20261123 + idx)
    tensors = {}
    for module, (o, i) in shapes.items():
        tensors["base_model.model.%s.lora_A.weight" % module] = (torch.randn(rank, i, generator=gen) * std).to(torch.bfloat16)
        tensors["base_model.model.%s.lora_B.weight" % module] = (torch.randn(o, rank, generator=gen) * std).to(torch.bfloat16)
    f = os.path.join(d, "adapter_model.safetensors")
    save_file(tensors, f)
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": alpha, "lora_dropout": 0.0,
           "bias": "none", "target_modules": suffixes, "inference_mode": True,
           "base_model_name_or_path": "RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8"}
    with open(os.path.join(d, "adapter_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    size = os.path.getsize(f)
    summary["adapters"].append({"name": name, "bytes": size, "sha256": h.hexdigest()})
    print(name, size, h.hexdigest()[:16])
with open(os.path.join(out_dir, "adapters-manifest.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
