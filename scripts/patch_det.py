import os, sys
p = os.path.expanduser("~/kv/venv034/lib/python3.12/site-packages/vllm/lora/ops/triton_ops/lora_shrink_op.py")
s = open(p).read()
tag = "KV_LORA_DET"
if tag in s:
    print("already patched"); sys.exit(0)
anchor = "_BI_SPLIT_K = 8\n"
assert s.count(anchor) == 1
add = ('# local test switch: fixed-order shrink reduce for the LoRA op only, global batch-invariant mode off\n'
       'is_batch_invariant = is_batch_invariant or __import__("os").environ.get("KV_LORA_DET") == "1"\n')
open(p + ".pre-det", "w").write(s)
open(p, "w").write(s.replace(anchor, anchor + add))
print("patched")
