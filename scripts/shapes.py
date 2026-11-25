"""Derive the adapter-kernel call shapes of dense decoder models from their Hugging Face config.json.
vLLM packs q/k/v into one 3-slice projection and gate/up into one 2-slice projection; o_proj and down_proj are single-slice."""
import json, sys, urllib.request
MODELS = ["unsloth/Llama-3.2-3B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B",
          "unsloth/mistral-7b-instruct-v0.3", "microsoft/phi-4", "unsloth/gemma-3-27b-it", "Qwen/Qwen3-32B",
          "Qwen/Qwen2.5-72B-Instruct", "unsloth/Llama-3.3-70B-Instruct"]
out = {}
for m in MODELS:
    try:
        c = json.load(urllib.request.urlopen("https://huggingface.co/%s/resolve/main/config.json" % m, timeout=30))
    except Exception as e:
        print("skip", m, e); continue
    c = c.get("text_config", c)
    h, inter, nh = c["hidden_size"], c["intermediate_size"], c["num_attention_heads"]
    nkv = c.get("num_key_value_heads", nh); hd = c.get("head_dim") or h // nh; L = c["num_hidden_layers"]
    q, kv = nh * hd, nkv * hd
    out[m] = {"layers": L, "hidden": h, "calls": [
        {"name": "qkv", "shrink_k": h, "expand": [q, kv, kv]},
        {"name": "o_proj", "shrink_k": q, "expand": [h]},
        {"name": "gate_up", "shrink_k": h, "expand": [inter, inter]},
        {"name": "down", "shrink_k": inter, "expand": [h]}]}
    print(m, "L", L, "hidden", h, "inter", inter, "q", q, "kv", kv)
out["Nemotron-3.5-Super-120B (measured)"] = {"layers": 0, "hidden": 4096, "calls": [
    {"name": "mamba_in_proj", "shrink_k": 4096, "expand": [8192, 8192, 1024, 1024, 128]},
    {"name": "attn_qkv", "shrink_k": 4096, "expand": [4096, 256, 256]},
    {"name": "single_4096", "shrink_k": 4096, "expand": [4096]}, {"name": "single_5376", "shrink_k": 5376, "expand": [5376]},
    {"name": "single_8192_256", "shrink_k": 8192, "expand": [256]}]}
json.dump(out, open(sys.argv[1], "w"), indent=1)
