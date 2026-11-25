import json,sys,os,csv
for r in sys.argv[1:]:
    try: s=json.load(open(r+"/client-summary.json"))
    except Exception as e: print(os.path.basename(r),"NO SUMMARY"); continue
    g=[x for x in csv.reader(open(r+"/gpu-telemetry.csv"))][1:]
    mem=[float(x[1]) for x in g if len(x)>4]; pw=[float(x[4]) for x in g if len(x)>4]
    cpu=sorted(float(l.split()[2]) for l in open(r+"/cpu-threads.txt")) if os.path.exists(r+"/cpu-threads.txt") else []
    cpu=cpu or [0]
    print("%-34s ok %d/%d lat p50 %.2f p95 %.2f ttft p50 %.3f p95 %.2f itl %.1fms tok/s %.0f req/s %.3f mem %d W %.0f cpu %.0f/%.0f" % (os.path.basename(r), s["valid_requests"], s["requests"], s["request_latency_seconds"]["p50"], s["request_latency_seconds"]["p95"], s["ttft_seconds"]["p50"], s["ttft_seconds"]["p95"], 1000*s["mean_inter_token_latency_seconds_per_request"]["p50"], s["output_tokens_per_second"], s["requests_per_second"], max(mem), sum(pw)/len(pw), cpu[len(cpu)//2], cpu[-1]))
