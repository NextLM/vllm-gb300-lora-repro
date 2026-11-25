"""Run-to-run determinism probe, modelled on the probe in vllm issue 50602: byte-identical requests, greedy decoding,
full generations, byte-compare the outputs. Public prompts only: the text of Python 3.12 standard-library modules.
usage: dprobe.py <out.json> <repeats> <max_tokens> <model> [<model> ...]
For each model and prompt: <repeats> identical requests one after another, then <repeats> sent together. Stores hashes and token counts, not text."""
import concurrent.futures, hashlib, json, sys, time, urllib.request, sysconfig, os
BASE = "http://127.0.0.1:8000"; out, repeats, max_tokens, models = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
lib = sysconfig.get_paths()["stdlib"]
def src(name, chars): return open(os.path.join(lib, name), encoding="utf-8").read()[:chars]
PROMPTS = {"long-argparse-30k": "Summarize what this Python module does in one paragraph.\n\n" + src("argparse.py", 30000),
           "long-subprocess-30k": "List the public functions defined in this Python module and say what each is for.\n\n" + src("subprocess.py", 30000),
           "mid-textwrap-8k": "Explain how this Python module wraps text.\n\n" + src("textwrap.py", 8000),
           "short": "Explain in three sentences why floating-point addition is not associative."}
def call(model, prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens, "seed": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900)); m = r["choices"][0]["message"]
    text = (m.get("reasoning_content") or m.get("reasoning") or "") + "\x00" + (m.get("content") or "")
    return hashlib.sha256(text.encode()).hexdigest()[:16], r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]
res = []
for model in models:
    for pname, prompt in PROMPTS.items():
        for mode in ("sequential", "concurrent"):
            t = time.time()
            if mode == "sequential": rs = [call(model, prompt) for _ in range(repeats)]
            else:
                with concurrent.futures.ThreadPoolExecutor(repeats) as ex: rs = list(ex.map(lambda _: call(model, prompt), range(repeats)))
            hashes = [r[0] for r in rs]; counts = {}
            for h in hashes: counts[h] = counts.get(h, 0) + 1
            rec = {"model": model, "prompt": pname, "mode": mode, "repeats": repeats, "prompt_tokens": rs[0][1], "completion_tokens": sorted({r[2] for r in rs}),
                   "distinct_outputs": len(counts), "largest_group": max(counts.values()), "hashes": hashes, "seconds": round(time.time() - t, 1)}
            res.append(rec); json.dump(res, open(out, "w"), indent=1)
            print("%-22s %-20s %-10s prompt %5d tok  distinct %d/%d  (largest group %d)" % (model, pname, mode, rec["prompt_tokens"], rec["distinct_outputs"], repeats, rec["largest_group"]), flush=True)
