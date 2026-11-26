@BeheSQ thanks, that's a clean result, and the zeroed-adapter control is a nice touch. Good to have it confirmed on GB10 with a different checkpoint and a trained adapter.

In case it's useful: jeejeelee opened #70197, which adds `--lora-split-k-reduction kernel`. It is a different route from your config file: the split stays on and the partial sums are added in a fixed order, without global batch-invariant mode. I'm testing it on GB300 now. If you have a chance to run it on your GB10 setup, a second hardware data point would be useful there.
