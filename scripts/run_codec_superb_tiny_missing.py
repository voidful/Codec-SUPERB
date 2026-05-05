import argparse
import glob
import json
import os
import shutil

from SoundCodec.codec import list_codec
from scripts.benchmarking import evaluate_dataset


def is_completed_result(metrics):
    if not isinstance(metrics, dict) or metrics.get("error"):
        return False
    if metrics.get("encode_only"):
        return True
    return any(
        category != "audio_samples" and isinstance(values, dict) and bool(values)
        for category, values in metrics.items()
    )


def collect_completed_models():
    completed = set()
    for path in sorted(glob.glob("*codec-superb-tiny*evaluation_results*.json")):
        with open(path) as f:
            data = json.load(f)
        for model, metrics in data.items():
            if is_completed_result(metrics):
                completed.add(model)
    return completed


def main():
    parser = argparse.ArgumentParser(description="Run missing codec-superb-tiny codec evaluations.")
    parser.add_argument("--dataset", default="voidful/codec-superb-tiny")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--max_duration", type=int, default=120)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_suffix", default=None)
    parser.add_argument("--keep_cache", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    completed = collect_completed_models()
    if args.models:
        models = [model for model in args.models if model not in completed]
    else:
        models = sorted(set(list_codec()) - completed)

    if args.num_shards > 1:
        models = [model for idx, model in enumerate(models) if idx % args.num_shards == args.shard_index]

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = "cache_original" if args.num_shards == 1 else f"cache_original_shard{args.shard_index}"

    output_suffix = args.output_suffix
    if output_suffix is None and args.num_shards > 1:
        output_suffix = f"shard{args.shard_index}"

    thread_limit = os.environ.get("CODEC_SUPERB_THREAD_LIMIT", "1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, thread_limit)

    print(f"Models to evaluate ({len(models)}): {models}", flush=True)
    print(f"max_workers={args.max_workers}, chunksize={args.chunksize}, cache_dir={cache_dir}, output_suffix={output_suffix}", flush=True)
    if not models:
        return

    try:
        evaluate_dataset(
            args.dataset,
            False,
            models,
            max_duration=args.max_duration,
            max_workers=args.max_workers,
            chunksize=args.chunksize,
            limit=None,
            save_audio=False,
            disk_cache=True,
            cleanup_cache=not args.keep_cache,
            cache_dir=cache_dir,
            output_suffix=output_suffix,
        )
    finally:
        if not args.keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
