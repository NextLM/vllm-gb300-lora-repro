"""Transfer benchmark for vLLM's lora_shrink / lora_expand kernels on this GPU.
For each call shape of each model x rank x max_loras x batch size, time these arms (CUDA graph of 8 calls, replayed; best of 3 rounds):
  default  : vLLM's built-in config
  cap      : default with split_k capped at the number of K blocks (the change proposed in vllm PR 61087)
  ours     : our Nemotron speed-tuned GB300 file, resolved with vLLM's own nearest-key lookup
  nosplit  : our Nemotron no-split GB300 file, same lookup
  oracle   : best of a small search on this exact shape (only with --oracle)
Each arm's output is checked against an fp32 reference computed with plain torch.
usage: xfer.py shapes.json out.jsonl <speed-config-dir> <nosplit-config-dir> [--oracle] [--models substr,substr] [--quick]"""
import itertools, json, math, sys, time, torch
import vllm.lora.ops.triton_ops.lora_shrink_op as S
import vllm.lora.ops.triton_ops.lora_expand_op as E
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
from vllm.lora.ops.triton_ops.utils import get_lora_op_configs as real_cfg
import vllm.lora.ops.triton_ops.utils as U   # its pointer caches are keyed by tensor address; clear them per case because this script frees and reallocates weights
dev = "cuda"; REPS = 8; CUR = {}
S.get_lora_op_configs = lambda *a, **k: dict(CUR); E.get_lora_op_configs = lambda *a, **k: dict(CUR)
shapes = json.load(open(sys.argv[1])); outp = sys.argv[2]
name = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
FILES = {arm: {"shrink": json.load(open("%s/%s_SHRINK.json" % (d, name))), "expand": json.load(open("%s/%s_EXPAND_TRUE.json" % (d, name)))} for arm, d in (("ours", sys.argv[3]), ("nosplit", sys.argv[4]))}
ORACLE = "--oracle" in sys.argv; QUICK = "--quick" in sys.argv
only = sys.argv[sys.argv.index("--models") + 1].split(",") if "--models" in sys.argv else None
RANKS = [16, 64] if QUICK else [8, 16, 64, 256]; LORAS = [4] if QUICK else [1, 4, 16]; MS = [1, 64] if QUICK else [1, 8, 64, 256, 2048]

def near(d, v): return d.get(str(v)) or d[min(d.keys(), key=lambda x: abs(int(x) - v))]
def lookup(tbl, max_loras_p1, slices, m, k, n):   # vLLM's lookup; slice count also falls back to nearest (upstream raises KeyError)
    d = near(tbl, max_loras_p1); d = near(d, slices); d = near(d, m); d = near(d, k); return dict(near(d, n))
def default_cfg(op, ml, M, hidden, rank, slices):
    return dict(real_cfg(op, max_loras=ml + 1, batch=M, hidden_size=hidden, rank=rank, num_slices=slices, add_inputs=True if op == "expand" else None))

def setup(op, M, rank, ml, k, widths):
    meta = LoRAKernelMeta.make(ml, M, dev); mapping = (torch.arange(M, device=dev) % ml).to(torch.int32)
    meta.prepare_tensors(mapping); margs = meta.meta_args(M, False); g = torch.Generator(device=dev).manual_seed(0); ns = len(widths)
    if op == "shrink":
        x = torch.randn(M, k, device=dev, generator=g).to(torch.bfloat16)
        w = [(torch.randn(ml, rank, k, device=dev, generator=g) * 0.02).to(torch.bfloat16) for _ in range(ns)]
        y = torch.zeros(ns, M, rank, device=dev, dtype=torch.float32)
        ref = torch.stack([0.5 * torch.einsum("mk,mrk->mr", x.float(), w[s].float()[mapping.long()]) for s in range(ns)])
        return (lambda: S._lora_shrink(x, w, y, *margs, 0.5)), y, ref
    x = torch.randn(ns, M, rank, device=dev, generator=g, dtype=torch.float32)
    w = [(torch.randn(ml, wd, rank, device=dev, generator=g) * 0.02).to(torch.bfloat16) for wd in widths]
    y = torch.zeros(M, sum(widths), device=dev, dtype=torch.bfloat16)
    ref = torch.cat([torch.einsum("mr,mwr->mw", x[s], w[s].float()[mapping.long()]) for s in range(ns)], dim=1)
    return (lambda: E._lora_expand(x, w, y, *margs, offset_start=0, add_inputs=True)), y, ref

def run_once(fn, y, cfg):
    CUR.clear(); CUR.update(cfg); y.zero_(); fn(); torch.cuda.synchronize()
def timed(fn, y, cfg, replays=30, rounds=3):
    run_once(fn, y, cfg); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPS): fn()
    for _ in range(3): g.replay()
    best = 1e18
    for _ in range(rounds):
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(replays): g.replay()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter() - t) / (replays * REPS) * 1e6)
    return best
def oracle_grid(op, M):
    bms = [b for b in (16, 32, 64) if b <= max(16, M * 2)]
    if op == "shrink":
        for bm, bn, bk, sk, nw in itertools.product(bms, [16, 32, 128, 256], [64, 256], [1, 8], [4, 8]):
            yield {"block_m": bm, "block_n": bn, "block_k": bk, "split_k": sk, "num_warps": nw, "num_ctas": 1, "group_size_m": 8, "num_stages": 2, "max_nreg": None}
    else:
        for bm, bn, bk, nw in itertools.product(bms, [32, 64, 128, 256], [32, 64], [4, 8]):
            yield {"block_m": bm, "block_n": bn, "block_k": bk, "num_warps": nw, "num_ctas": 1, "num_stages": 2, "max_nreg": None}

seen = set(); fo = open(outp, "a"); n = 0; t0 = time.time()
for model, spec in shapes.items():
    if only and not any(o.lower() in model.lower() for o in only): continue
    for call in spec["calls"]:
        widths = call["expand"]; ns = len(widths)
        for op, rank, ml, M in itertools.product(("shrink", "expand"), RANKS, LORAS, MS):
            key = (op, call["shrink_k"] if op == "shrink" else tuple(widths), ns, rank, ml, M)
            rec = {"model": model, "call": call["name"], "op": op, "slices": ns, "rank": rank, "max_loras": ml, "M": M, "k": call["shrink_k"], "widths": widths, "dup": key in seen}
            if key in seen:
                fo.write(json.dumps(rec) + "\n"); continue
            seen.add(key)
            hidden = call["shrink_k"] if op == "shrink" else max(widths)
            try:
                U._LORA_A_PTR_DICT.clear(); U._LORA_B_PTR_DICT.clear()
                fn, y, ref = setup(op, M, rank, ml, call["shrink_k"], widths)
                d = default_cfg(op, ml, M, hidden, rank, ns); arms = {"default": d}
                if op == "shrink":
                    arms["cap"] = {**d, "split_k": max(1, min(d["split_k"], math.ceil(call["shrink_k"] / d["block_k"])))}
                    for a in ("ours", "nosplit"): arms[a] = lookup(FILES[a]["shrink"], ml + 1, ns, M, call["shrink_k"], rank)
                else:
                    for a in ("ours", "nosplit"): arms[a] = lookup(FILES[a]["expand"], ml + 1, ns, M, rank, hidden)
                rec["arms"] = {}
                for a, cfg in arms.items():
                    try:
                        us = timed(fn, y, cfg); run_once(fn, y, cfg)
                        err = ((y.float() - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
                        rec["arms"][a] = {"us": round(us, 2), "err": err, "cfg": cfg}
                    except Exception as e:
                        rec["arms"][a] = {"us": None, "error": str(e)[:120], "cfg": cfg}
                if ORACLE:
                    best = None
                    for cfg in oracle_grid(op, M):
                        try:
                            run_once(fn, y, cfg); err = ((y.float() - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
                            if not err < 0.02: continue
                            us = timed(fn, y, cfg, replays=10, rounds=1)
                            if best is None or us < best[0]: best = (us, cfg, err)
                        except Exception: pass
                    if best:
                        us = timed(fn, y, best[1]); rec["arms"]["oracle"] = {"us": round(us, 2), "err": best[2], "cfg": best[1]}
                del fn, y, ref
            except Exception as e:
                rec["error"] = str(e)[:200]
            fo.write(json.dumps(rec) + "\n"); fo.flush(); n += 1
            if n % 50 == 0: print(n, "cases", round(time.time() - t0), "s", flush=True)
print("DONE", n, round(time.time() - t0), flush=True)
