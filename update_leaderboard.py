import json

import glob
import os

# Load latest benchmark results
json_files = glob.glob('*codec-superb-tiny_synth_evaluation_results*.json')
if not json_files:
    raise FileNotFoundError("No benchmark results found.")
latest_file = max(json_files, key=os.path.getmtime)
print(f"Loading results from {latest_file}")
with open(latest_file, 'r') as f:
    benchmark_results = json.load(f)

# Hardcoded BPS mapping (bitrate in kbps or as used in data.js)
bps_mapping = {
    'academicodec_hifi_16k_320d': 2,
    'academicodec_hifi_16k_320d_large_uni': 2,
    'academicodec_hifi_24k_320d': 3,
    'audiodec_24k_320d': 6.4,
    'auv': 1, # Estimated or placeholder
    'bigcodec_1k': 1, # Estimated or placeholder
    'dac_16k': 6,
    'dac_24k': 24,
    'dac_44k': 8,
    'encodec_24k_12bps': 12,
    'encodec_24k_1_5bps': 1.5,
    'encodec_24k_24bps': 24,
    'encodec_24k_3bps': 3,
    'encodec_24k_6bps': 6,
    'funcodec_en_libritts_16k_gr1nq32ds320': 16,
    'funcodec_en_libritts_16k_gr8nq32ds320': 16,
    'funcodec_en_libritts_16k_nq32ds320': 16,
    'funcodec_en_libritts_16k_nq32ds640': 8,
    'funcodec_zh_en_16k_nq32ds320': 16,
    'funcodec_zh_en_16k_nq32ds640': 8,
    's3tokenizer_v1': 0.1, # Semantic tokenizer
    'speech_tokenizer_16k': 4,
    'sqcodec_16k_0k75bps': 0.75,
    'sqcodec_16k_12kbps': 12,
    'sqcodec_16k_1k5bps': 1.5,
    'sqcodec_16k_3kbps': 3,
    'sqcodec_16k_6kbps': 6,
    'sqcodec_24k_12kbps': 12,
    'sqcodec_24k_24kbps': 24,
    'unicodec_24k': 12, # Estimated
    'wavtokenizer_24k_small_600_4096': 0.1,
    'wavtokenizer_24k_medium_600_4096': 0.1,
    'wavtokenizer_24k_large_600_4096': 0.1,
    'wavtokenizer_24k_large_speech_75token': 0.1
}

# Metrics to include
metrics_to_include = ['mel', 'pesq', 'stoi', 'f0corr']

new_results = {}

new_results = {}

for model_name, metrics_data in benchmark_results.items():
    entry = {
        'bps': bps_mapping.get(model_name, 0)
    }
    
    # Check if nested by category
    is_nested = any(isinstance(v, dict) for v in metrics_data.values())
    
    if is_nested:
        for category, metrics in metrics_data.items():
            for m in metrics_to_include:
                val = metrics.get(m, 0)
                if val != val: # NaN check
                    val = 0
                entry[f"{category.lower()}_{m}"] = round(float(val), 3)
    else:
        # Legacy format: metrics_data is {metric: value}
        # Map to 'overall' category by default
        for m in metrics_to_include:
            val = metrics_data.get(m, 0)
            if val != val: # NaN check
                val = 0
            entry[f"overall_{m}"] = round(float(val), 3)
            
    new_results[model_name] = entry

# Format as JavaScript
js_content = "const results = " + json.dumps(new_results, indent=1) + ";\nexport default results;"

with open('web/src/results/data.js', 'w') as f:
    f.write(js_content)

print(f"Updated web/src/results/data.js with {len(new_results)} codecs.")
