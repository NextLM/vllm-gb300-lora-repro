"""Streaming load client: TTFT, inter-token latency, request latency, tokens/s, windowed throughput.

Modes
  ctx <out_dir> <workers> <rounds> <prompt_tokens> <output_tokens> <model> [<model> ...]
      Full-context test. Each worker sends <rounds> requests; every request has a distinct
      synthetic prompt of exactly <prompt_tokens> tokens and is forced to generate exactly
      <output_tokens> tokens (ignore_eos). Worker w uses models[w % len(models)].

Outputs (no prompt or response text): requests-timing.jsonl, server-samples.csv, client-summary.json.
"""
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
mode, out_dir, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out_dir, exist_ok=True)
lock = threading.Lock()
results, gaps_per_token = [], []
stop_sampling = threading.Event()
T0 = time.monotonic()


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def sampler():
    """Poll server metrics once per second: running/waiting requests and KV cache usage."""
    with open(os.path.join(out_dir, "server-samples.csv"), "w") as fh:
        fh.write("offset_seconds,requests_running,requests_waiting,kv_cache_usage_fraction\n")
        while not stop_sampling.is_set():
            vals = {"run": None, "wait": None, "kv": None}
            try:
                for line in urllib.request.urlopen(BASE + "/metrics", timeout=5).read().decode().splitlines():
                    if line.startswith("vllm:num_requests_running"):
                        vals["run"] = float(line.rsplit(" ", 1)[1])
                    elif line.startswith("vllm:num_requests_waiting"):
                        vals["wait"] = float(line.rsplit(" ", 1)[1])
                    elif line.startswith("vllm:kv_cache_usage_perc") or line.startswith("vllm:gpu_cache_usage_perc"):
                        vals["kv"] = float(line.rsplit(" ", 1)[1])
            except Exception:  # noqa: BLE001
                pass
            fh.write("%.2f,%s,%s,%s\n" % (time.monotonic() - T0, vals["run"], vals["wait"], vals["kv"]))
            fh.flush()
            stop_sampling.wait(1.0)


def stream_once(body):
    """Send one streaming request. Returns timing dict (no text kept beyond a hash)."""
    body = dict(body, stream=True, stream_options={"include_usage": True, "continuous_usage_stats": True})
    started = time.monotonic()
    first, last, prev_t, prev_tok = None, None, None, 0
    usage, parts, local_gaps = {}, [], []
    with post("/v1/chat/completions", body) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            piece = None
            for ch in chunk.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or piece
            if piece:
                now = time.monotonic()
                tok = (usage or {}).get("completion_tokens") or (prev_tok + 1)
                if first is None:
                    first = now
                else:
                    dtok = max(tok - prev_tok, 1)
                    local_gaps.append((now - prev_t) / dtok)
                prev_t, prev_tok, last = now, tok, now
                parts.append(piece)
    ended = time.monotonic()
    ctoks = usage.get("completion_tokens") or 0
    itl = (last - first) / (ctoks - 1) if first is not None and ctoks > 1 else None
    return {
        "start_offset_seconds": started - T0, "end_offset_seconds": ended - T0,
        "ttft_seconds": (first - started) if first is not None else None,
        "elapsed_seconds": ended - started, "mean_itl_seconds": itl,
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": ctoks,
        "output_sha256_12": hashlib.sha256("".join(parts).encode()).hexdigest()[:12],
    }, "".join(parts), local_gaps


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def dist(v):
    v = [x for x in v if x is not None]
    return {"p50": pct(v, .5), "p95": pct(v, .95), "p99": pct(v, .99), "mean": sum(v) / len(v) if v else None,
            "max": max(v) if v else None, "n": len(v)}


def record(rec, gaps, total):
    with lock:
        results.append(rec)
        gaps_per_token.extend(gaps)
        if len(results) % 100 == 0 or len(results) == total:
            print("[%d/%d] %.0fs" % (len(results), total, time.monotonic() - T0), flush=True)


extra = {}
if mode == "ctx":
    rounds, prompt_tokens, output_tokens, models = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7:]
    total = workers * rounds
    words = ("river stone amber window quiet ledger harbor copper signal meadow winter anchor violet timber "
             "candle orbit pepper garden silver thunder marble canvas lantern prairie velvet compass echo "
             "summit basil cobalt drift ember fable glacier hollow ivory jasmine kettle linen mosaic nectar").split()

    def build_prompt(seed):
        """Distinct word-salad prompt whose chat-templated length is exactly prompt_tokens."""
        rng = random.Random(seed)
        seq = [rng.choice(words) for _ in range(prompt_tokens + 2000)]
        def count(text):
            msgs = [{"role": "user", "content": text}]
            got = json.loads(post("/tokenize", {"model": models[0], "messages": msgs, "add_generation_prompt": True,
                                                "chat_template_kwargs": {"enable_thinking": False}}).read())["count"]
            return msgs, got

        head = "Read the words, then keep writing more words.\n"
        n = prompt_tokens - 64
        got = 0
        for _ in range(400):
            text = head + " ".join(seq[:n])
            msgs, got = count(text)
            if got == prompt_tokens:
                return msgs
            short = prompt_tokens - got
            if short < 0:  # over: drop words (always moves, so no oscillation)
                n -= max(1, int(-short / 1.3))
            elif short > 40:  # far under: add words
                n = min(len(seq), n + int(short / 1.3))
            else:  # close and under: finish with single-token pads, else drop a word and retry
                for pad in (" a", " 1", ".", " x", ",", " the", "!"):
                    msgs, got2 = count(text + pad * short)
                    if got2 == prompt_tokens:
                        return msgs
                n -= 1
        raise RuntimeError("could not hit %d prompt tokens (last %d)" % (prompt_tokens, got))

    print("building %d distinct prompts of %d tokens ..." % (total, prompt_tokens), flush=True)
    prompts = [build_prompt(900000 + i) for i in range(total)]
    print("prompts built in %.0fs" % (time.monotonic() - T0), flush=True)
    T0 = time.monotonic()
    extra = {"target_prompt_tokens": prompt_tokens, "target_output_tokens": output_tokens, "rounds": rounds}

    def job(w):
        model = models[w % len(models)]
        for r in range(rounds):
            body = {"model": model, "messages": prompts[w * rounds + r], "temperature": 0, "seed": 1,
                    "max_tokens": output_tokens, "min_tokens": output_tokens, "ignore_eos": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            rec = {"request_id": "ctx-w%02d-r%02d" % (w, r), "worker": w, "round": r, "model": model}
            gaps, err = [], None
            try:
                timing, _, gaps = stream_once(body)
                rec.update(timing)
            except Exception as e:  # noqa: BLE001
                detail = e.read()[:300].decode("utf-8", "replace") if hasattr(e, "read") else str(e)[:200]
                err = "%s: %s" % (type(e).__name__, detail)
            rec.update({"valid": err is None, "error": err})
            record(rec, gaps, total)

    jobs = list(range(workers))
else:
    raise SystemExit("unknown mode")

threading.Thread(target=sampler, daemon=True).start()
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
    list(ex.map(job, jobs))
wall = time.monotonic() - T0
stop_sampling.set()
time.sleep(1.2)

results.sort(key=lambda r: r.get("start_offset_seconds") or 0)
with open(os.path.join(out_dir, "requests-timing.jsonl"), "w") as fh:
    for r in results:
        fh.write(json.dumps(r, sort_keys=True) + "\n")
ok = [r for r in results if r["valid"]]

# Peak number of client requests in flight at once (from client timestamps).
events = sorted([(r["start_offset_seconds"], 1) for r in ok] + [(r["end_offset_seconds"], -1) for r in ok])
inflight = peak = 0
for _, d in events:
    inflight += d
    peak = max(peak, inflight)

# Sustained throughput: completed requests and output tokens per full 60 s window.
win = {}
for r in ok:
    k = int(r["end_offset_seconds"] // 60)
    w = win.setdefault(k, [0, 0, 0])
    w[0] += 1; w[1] += r["completion_tokens"] or 0; w[2] += r["prompt_tokens"] or 0
full = [win.get(k, [0, 0, 0]) for k in range(int(wall // 60))]
rps = [w[0] / 60 for w in full]
ops = [w[1] / 60 for w in full]

samples = []
try:
    for line in list(open(os.path.join(out_dir, "server-samples.csv")))[1:]:
        p = line.strip().split(",")
        samples.append([None if x == "None" else float(x) for x in p])
except FileNotFoundError:
    pass
running = [x[1] for x in samples if x[1] is not None]
kv = [x[3] for x in samples if x[3] is not None]

ptoks = sum(r["prompt_tokens"] or 0 for r in ok)
ctoks = sum(r["completion_tokens"] or 0 for r in ok)
summary = dict(extra, **{
    "mode": mode, "workers": workers, "models": models,
    "requests": len(results), "valid_requests": len(ok), "failed_requests": len(results) - len(ok),
    "requests_with_retry": sum(1 for r in results if r.get("attempts", 1) > 1),
    "errors_sample": sorted({r["error"] for r in results if r["error"]})[:5],
    "wall_seconds": wall,
    "requests_per_second": len(ok) / wall,
    "predictions_per_second": sum(r.get("predictions", 0) for r in ok) / wall,
    "input_tokens": ptoks, "output_tokens": ctoks,
    "input_tokens_per_second": ptoks / wall, "output_tokens_per_second": ctoks / wall,
    "request_latency_seconds": dist([r["elapsed_seconds"] for r in ok]),
    "ttft_seconds": dist([r["ttft_seconds"] for r in ok]),
    "mean_inter_token_latency_seconds_per_request": dist([r["mean_itl_seconds"] for r in ok]),
    "inter_token_gap_seconds_pooled_chunks": dist(gaps_per_token),
    "prompt_tokens_per_request": {"min": min((r["prompt_tokens"] or 0) for r in ok) if ok else None,
                                  "max": max((r["prompt_tokens"] or 0) for r in ok) if ok else None},
    "completion_tokens_per_request": {"min": min((r["completion_tokens"] or 0) for r in ok) if ok else None,
                                      "max": max((r["completion_tokens"] or 0) for r in ok) if ok else None},
    "client_peak_requests_in_flight": peak,
    "server_requests_running": {"max": max(running) if running else None,
                                "fraction_of_samples_at_or_above_workers": (sum(1 for x in running if x >= workers) / len(running)) if running else None,
                                "samples": len(running)},
    "server_kv_cache_usage_fraction": {"max": max(kv) if kv else None, "mean": sum(kv) / len(kv) if kv else None},
    "sustained_60s_windows": {"full_windows": len(full),
                              "requests_per_second_min_median_max": [min(rps), pct(rps, .5), max(rps)] if rps else None,
                              "output_tokens_per_second_min_median_max": [min(ops), pct(ops, .5), max(ops)] if ops else None},
})
json.dump(summary, open(os.path.join(out_dir, "client-summary.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=1))
