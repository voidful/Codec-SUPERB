import argparse
import json
import gc
import os
import time
import soundfile as sf
import io

# The previous hacks are removed as they were ineffective against datasets behavior.


def default_converter(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def safe_load_audio(audio_entry):
    """
    Safely load audio array and sampling rate, bypassing torchcodec if simple path reading is possible.
    Returns: (array [C, T] or [T], sampling_rate) or (None, None) if failed.
    """
    try:
        # Try to extract path first to use soundfile directly
        path = None
        if isinstance(audio_entry, dict):
            if 'path' in audio_entry and audio_entry['path'] is not None:
                path = audio_entry['path']
        elif hasattr(audio_entry, "__getitem__"):
             # For AudioDecoder, try to access 'path' without triggering full decode if possible.
             # But usually accessing any key decodes everything in newer datasets versions.
             # However, accessing 'path' specifically *might* be safe or exposed as property.
             try:
                 path = audio_entry['path']
             except:
                 pass
        
        if path:
            try:
                # Load with soundfile
                array, sr = sf.read(path)
                return array, sr
            except Exception:
                # Fallback if soundfile fails (e.g. mp3 without libs, or remote URL)
                pass
        
        # Fallback to accessing 'array' which triggers datasets decoding (and potentially torchcodec)
        if hasattr(audio_entry, "__getitem__") or isinstance(audio_entry, dict):
             if 'array' in audio_entry and 'sampling_rate' in audio_entry:
                 return audio_entry['array'], audio_entry['sampling_rate']

        return None, None
    except Exception as e:
        print(f"Error in safe_load_audio: {e}")
        return None, None


def compute_metrics(original, model, max_duration):
    orig_array, orig_sr = safe_load_audio(original['audio'])
    model_array, model_sr = safe_load_audio(model['audio'])
    
    if orig_array is None or model_array is None:
        return None # Skip if audio loading failed

    original_arrays, resynth_array = pad_arrays_to_match(orig_array, model_array)
    # Use the SR from original (should match model ideally, or be handled by AudioSignal)
    # AudioSignal handles resampling if needed, but here we assume match after padding or logic in pad_arrays_to_match
    
    # Check if pad_arrays_to_match worked (it returns numpy arrays)
    
    sampling_rate = orig_sr
    original_signal = AudioSignal(original_arrays, sampling_rate)
    if original_signal.duration > max_duration:
        return None
    model_signal = AudioSignal(resynth_array, sampling_rate)
    metrics = get_metrics(original_signal, model_signal)
    return metrics


def process_entry(args):
    original_iter, model_iter, max_duration = args
    try:
        metrics = compute_metrics(original_iter, model_iter, max_duration)
        if metrics is not None:
            return metrics
        else:
            return {}
    except Exception as e:
        import traceback
        print(f"Error processing entry: {e}")
        try:
            print("Debug Info:")
            if isinstance(original_iter, dict):
                print(f"Original keys: {list(original_iter.keys())}")
                if 'audio' in original_iter:
                    audio_data = original_iter['audio']
                    print(f"Original audio type: {type(audio_data)}")
                    if isinstance(audio_data, dict) and 'array' in audio_data:
                        arr = audio_data['array']
                        print(f"Original audio array type: {type(arr)}")
                        if hasattr(arr, 'shape'):
                            print(f"Original audio array shape: {arr.shape}")
            
            if isinstance(model_iter, dict):
                print(f"Model keys: {list(model_iter.keys())}")
                if 'audio' in model_iter:
                    audio_data = model_iter['audio']
                    print(f"Model audio type: {type(audio_data)}")
                    if isinstance(audio_data, dict) and 'array' in audio_data:
                        arr = audio_data['array']
                        print(f"Model audio array type: {type(arr)}")
                        if hasattr(arr, 'shape'):
                            print(f"Model audio array shape: {arr.shape}")
            print("Traceback:")
            traceback.print_exc()
        except Exception as debug_e:
            print(f"Error printing debug info: {debug_e}")
            traceback.print_exc()
        return {}


def evaluate_dataset(dataset_name, is_stream, specific_models=None, max_duration=120, max_workers=4, chunksize=10, limit=None):
    start_time = time.time()  # Start time measurement
    print(f"Initial RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB\n")

    if os.path.exists(dataset_name):
        c = load_from_disk(dataset_name)
    else:
        c = load_dataset(dataset_name, streaming=is_stream)
    models = [key for key in c.keys() if key != "original"]

    result_data = {}
    for model in models:
        if specific_models is not None and model not in specific_models:
            continue
        print(f"Evaluating metrics for model: {model}")
        model_start_time = time.time()

        # Process Dataset with Multi-Processing
        args_list = [(original_iter, model_iter, max_duration) for original_iter, model_iter in
                     zip(c['original'], c[model])]
        
        if limit is not None:
            args_list = args_list[:limit]

        metrics_results = process_map(process_entry, args_list, max_workers=max_workers, chunksize=chunksize)
        metrics_results = [metrics for metrics in metrics_results if metrics is not None]
        # Process Dataset END

        # Aggregate the metrics
        aggregated_metrics = defaultdict(lambda: defaultdict(list))
        for metrics, entry in zip(metrics_results, c['original']):
            if metrics:
                category = entry.get('category', 'overall')
                for k, v in metrics.items():
                    aggregated_metrics[category][k].append(v)
                    aggregated_metrics['overall'][k].append(v)

        # Calculate average metrics per category
        model_result = {}
        for category, metrics_dict in aggregated_metrics.items():
            model_result[category] = {k: np.nanmean(v) if v else np.nan for k, v in metrics_dict.items()}
        
        result_data[model] = model_result
        gc.collect()
        print(f"RAM used after processing {model}: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        print(f"Time taken for {model}: {time.time() - model_start_time:.2f} seconds")
        print(model_result)
        print()

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"Final RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # Save results
    output_file_name = f"{dataset_name.replace('/', '_')}_evaluation_results.json"
    with open(output_file_name, 'w') as out_file:
        json.dump(result_data, out_file, indent=4, default=default_converter)

    base_filename = f"{args.dataset.replace('/', '_')}_evaluation_results"
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S") if os.path.exists(f"{base_filename}.json") else ""
    output_file_name = f"{base_filename}{timestamp}.json"

    # Save results to the file
    with open(output_file_name, 'w') as out_file:
        json.dump(result_data, out_file, indent=4, default=default_converter)

    print(f"Results saved to {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate audio datasets.')
    parser.add_argument('--dataset', type=str, default="AudioDecBenchmark/librispeech_asr_dummy_synth",
                        help='Name of the dataset to evaluate')
    parser.add_argument('--streaming', action='store_true', help='Evaluate in streaming mode')
    parser.add_argument('--batch', type=int, default=100,
                        help='Batch size for processing the dataset')
    parser.add_argument('--models', nargs='*', help='Specific models to evaluate')
    parser.add_argument('--max_duration', type=int, default=120,
                        help='Maximum duration of audio recordings in seconds')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of workers for multi-processing')
    parser.add_argument('--chunksize', type=int, default=10, help='Chunk size for multi-processing')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to evaluate')

    args = parser.parse_args()
    evaluate_dataset(args.dataset, args.streaming, args.models, args.max_duration, args.max_workers, args.chunksize, args.limit)
