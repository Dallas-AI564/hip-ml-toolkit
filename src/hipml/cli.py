"""CLI entry point for hip-ml-toolkit."""

import argparse
import sys
import json

from hipml import __version__
from hipml.device import get_device_info


def main():
    parser = argparse.ArgumentParser(
        prog="hipml",
        description="ML Inference Optimization Toolkit for AMD GPUs",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # ---- info ----
    sub.add_parser("info", help="Show AMD GPU information")

    # ---- serve ----
    p_serve = sub.add_parser("serve", help="Serve a model")
    p_serve.add_argument("model", help="Model name or path")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--quantize", choices=["fp8", "int4", "none"], default="none")
    p_serve.add_argument("--tensor-parallel", type=int, default=1)
    p_serve.add_argument("--max-batch-size", type=int, default=64)

    # ---- benchmark ----
    p_bench = sub.add_parser("benchmark", help="Benchmark a model")
    p_bench.add_argument("--model", required=True)
    p_bench.add_argument("--batch-size", default="1,4,8,16")
    p_bench.add_argument("--seq-len", type=int, default=2048)
    p_bench.add_argument("--iterations", type=int, default=100)
    p_bench.add_argument("--output", default="results/")

    # ---- profile ----
    p_prof = sub.add_parser("profile", help="Profile inference")
    p_prof.add_argument("--model", required=True)
    p_prof.add_argument("--output", default="profile.json")

    # ---- quantize ----
    p_quant = sub.add_parser("quantize", help="Quantize a model")
    p_quant.add_argument("--model", required=True)
    p_quant.add_argument("--method", choices=["gptq", "awq", "fp8"], default="fp8")
    p_quant.add_argument("--output", default="quantized_model/")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "info":
        info = get_device_info()
        print(json.dumps(info, indent=2))
    elif args.command == "serve":
        _serve(args)
    elif args.command == "benchmark":
        _benchmark(args)
    elif args.command == "profile":
        _profile(args)
    elif args.command == "quantize":
        _quantize(args)


def _serve(args):
    print(f"Serving {args.model} on port {args.port}")
    print(f"Quantize: {args.quantize}, TP: {args.tensor_parallel}")
    from hipml.serve import serve_model
    serve_model(args.model, port=args.port, quantize=args.quantize,
                tensor_parallel=args.tensor_parallel)


def _benchmark(args):
    batch_sizes = [int(x) for x in args.batch_size.split(",")]
    print(f"Benchmarking {args.model}")
    print(f"Batch sizes: {batch_sizes}, Seq len: {args.seq_len}")
    from hipml.benchmarks.throughput import run_throughput_benchmark
    results = run_throughput_benchmark(args.model, batch_sizes, args.seq_len, args.iterations)
    print(json.dumps(results, indent=2))


def _profile(args):
    print(f"Profiling {args.model}")
    from hipml.optimizers.profiler import profile_model
    results = profile_model(args.model)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Profile saved to {args.output}")


def _quantize(args):
    print(f"Quantizing {args.model} with {args.method}")
    from hipml.optimizers.quantizer import quantize_model
    quantize_model(args.model, method=args.method, output_dir=args.output)


if __name__ == "__main__":
    main()
