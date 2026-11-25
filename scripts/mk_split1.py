"""Write a shrink config file that equals vLLM's default shrink settings except split_k=1, for every shape (nearest-key lookup)."""
import json, os, sys, torch
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
def cfg(small): return {"block_m": 32, "block_n": 16, "block_k": 256 if small else 32, "split_k": 1, "num_warps": 4, "num_ctas": 1, "group_size_m": 8, "num_stages": 2, "max_nreg": None}
m = {"1": cfg(True), "127": cfg(True), "128": cfg(False), "1000000": cfg(False)}
tbl = {"1": {str(s): {mk: {"4096": {"8": c}} for mk, c in m.items()} for s in range(1, 9)}}
name = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
json.dump(tbl, open("%s/%s_SHRINK.json" % (out, name), "w"), indent=1); print("wrote", out, name)
