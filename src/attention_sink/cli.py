import argparse
from .viz import build_keep_mask_over_time, plot_keep_mask, live_generate_and_record

def main():
    ap = argparse.ArgumentParser(description="Attention Sinks toolkit")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", action="store_true", help="Visualization without HF model.")
    mode.add_argument("--live", action="store_true", help="Run HF model and visualize kept indices.")

    ap.add_argument("--sink_size", type=int, default=6)
    ap.add_argument("--window_size", type=int, default=768)

    # Simulation
    ap.add_argument("--total_steps", type=int, default=120)
    ap.add_argument("--seq_start", type=int, default=1024)

    # Live
    ap.add_argument("--model", type=str, default="google/gemma-2b-it")
    ap.add_argument("--prompt", type=str, default="Explain attention sinks in one paragraph.")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    args = ap.parse_args()

    if args.simulate:
        mask = build_keep_mask_over_time(args.total_steps, args.sink_size, args.window_size, args.seq_start)
        plot_keep_mask(mask, title=f"Simulation — sinks={args.sink_size}, window={args.window_size}, start={args.seq_start}")
    else:
        mask = live_generate_and_record(
            model_name=args.model,
            prompt=args.prompt,
            sink_size=args.sink_size,
            window_size=args.window_size,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            dtype_str=args.dtype,
        )
        plot_keep_mask(mask, title=f"Live ({args.model}) — sinks={args.sink_size}, window={args.window_size}")

if __name__ == "__main__":
    main()
