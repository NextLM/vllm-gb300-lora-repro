Thanks, Vladimir, that's a great data point, and the zeroed-adapter control again makes it airtight.

One clarification on the comparison: the −34% on GB300 was `split_k=1` with default block sizes. With `--lora-split-k-reduction kernel` I measured −7% on Llama-3.2-3B and no loss on Nemotron, so GB10 and GB300 line up: the fixed-order path is repeatable at a single-digit cost on both.

#70197 has since merged, so the flag is on main now, with coverage on both hardware families.
