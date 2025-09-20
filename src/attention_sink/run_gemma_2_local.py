from streaming_llm import StreamingGenerator
from viz import build_keep_mask_over_time, plot_keep_mask

gen = StreamingGenerator("google/gemma-2b-it", sink_size=6, window_size=768, device="mps")
print(gen.generate_stream("Explain attention sinks briefly:", max_new_tokens=120))


keep_mask = build_keep_mask_over_time(120, 6, 768, 1024)
plot_keep_mask(keep_mask, title="Attention Sinks Simulation")