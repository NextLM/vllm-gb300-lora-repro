"""Record the shrink/expand kernel settings vLLM resolves under the current environment, for the shapes of one model.
usage: logcfg.py <out.json> <max_loras> <rank> <shrink_k,slices;...> <expand_n,slices;...>   (run with the same env vars as the server)"""
import json, math, os, sys
from vllm.lora.ops.triton_ops.utils import get_lora_op_configs, is_batch_invariant
out, ml, rank = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); rows = []
for M in (1, 4, 16, 64, 127, 128, 256, 2048):
    for spec in sys.argv[4].split(";"):
        k, s = map(int, spec.split(",")); c = dict(get_lora_op_configs("shrink", max_loras=ml + 1, batch=M, hidden_size=k, rank=rank, num_slices=s))
        eff = c["split_k"]; bk = c["block_k"]
        if is_batch_invariant: eff, bk = 8, 256
        if os.environ.get("KV_CAP_SPLITK") == "1": eff = min(eff, math.ceil(k / bk))
        rows.append({"op": "shrink", "M": M, "k": k, "slices": s, "config": c, "effective_split_k": eff, "effective_block_k": bk, "k_blocks": math.ceil(k / bk)})
    for spec in sys.argv[5].split(";"):
        n, s = map(int, spec.split(",")); rows.append({"op": "expand", "M": M, "n": n, "slices": s, "config": dict(get_lora_op_configs("expand", max_loras=ml + 1, batch=M, hidden_size=n, rank=rank, num_slices=s, add_inputs=True))})
json.dump({"env": {k: v for k, v in os.environ.items() if k.startswith(("VLLM_TUNED", "VLLM_BATCH", "KV_"))}, "batch_invariant": bool(is_batch_invariant), "rows": rows}, open(out, "w"), indent=1)
print(out, {(r["M"], r["k"], r["slices"]): (r["effective_split_k"], r["k_blocks"]) for r in rows if r["op"] == "shrink" and r["M"] in (1, 64, 256)})
