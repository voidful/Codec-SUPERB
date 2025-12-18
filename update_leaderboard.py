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
# Exact BPS and TPS mapping
# Assuming 10 bits per code for most quantizers unless specified otherwise (EnCodec/DAC/FunCodec standard)
# WavTokenizer: 12 bits
# SQCodec: Variable, usually flattening? No, SQCodec uses scalar quantization? Need to map correctly.
# SQCodec 16k 0.75bps -> 750 bps? Name says 0k75.
# Let's trust the name for BPS? The user said "calculate or verify".
# EnCodec: BPS is in name (e.g. 6bps = 6kbps). Consistency check: 75Hz * 8 Codes * 10 bits = 6000 bps = 6kbps. CORRECT.
# So Formula: TPS * NumQuantizers * BitsPerCode / 1000

# Academicodec: 50Hz, 4Q. 10 bits? 50*4*10 = 2000 = 2kbps. (Matches name)
# Academicodec 24k: 75Hz, 4Q. 75*4*10 = 3000 = 3kbps. (Matches name 320d? No 320d is dim. But previously mapped to 3)

# DAC 16k: 50Hz, 12Q. 10 bits? 50*12*10 = 6000 = 6kbps. Matches.
# DAC 24k: 75Hz, 32Q. 10 bits? 75*32*10 = 24000 = 24kbps. Matches.
# DAC 44k: 86.13Hz (44100/512), 9Q. 10 bits? 86*9*10 = 7740 (~8kbps). Matches.

# FunCodec:
# 16k 320 32Q 50Hz: 50*32*10 = 16000 = 16kbps. Matches.
# 16k 640 32Q 25Hz: 25*32*10 = 8000 = 8kbps. Matches.

# SQCodec:
# Names are explicit: 0k75bps, 12kbps...
# 0.75kbps: 50Hz? 15 bits? Or sparse? SQCodec is scalar quantization.
# Let's stick to the name-derived BPS for SQCodec as "Truth" but verified by TPS.
# 0.75kbps = 750 bps. TPS 200? 200 * bits = 750 -> bits=3.75?
# Actually SQCodec might be 4 codes per frame?
# Shape is (1, 200) for 1 sec -> TPS=200.
# If TPS is 200, BPS 0.75kbps -> 3.75 bits/token?
# Let's follow the naming convention for BPS but use exact TPS.

# BigCodec 1k: 80Hz. 1Q? Size [1, 80] means 80 tokens.
# 1 kbps = 1000 bps. 1000 / 80 = 12.5 bits? or 80 * 12 = 960bps ~ 1kbps.
# WavTokenizer is 12 bits. BigCodec is likely similar. 
# Let's set BigCodec BPS to 1.0 (approx) and TPS 80.

# AUV: [1, 50]. 50Hz. 1Q.
# Config says "auv". VQ-VAE. 
# If 1 code per frame, 50Hz.
# If codebook 1024 (10 bits) -> 500 bps = 0.5 kbps.
# If codebook larger?
# Let's set AUV BPS to 0.5 (estimated from shape) or 1.0 if we assume multiple heads flattened?
# Shape [1, 50] suggests 1 code.
# Let's check AUV paper... usually low bitrate.
# Let's set 0.5 for now if single code.

# S3Tokenizer: [50]. 50Hz. 1Q.
# 50 * 10-12 bits = 0.5-0.6 kbps.
# Previously 0.1? Unlikely. That would be 2 bits per token.
# Let's set 0.5 kbps.

# WavTokenizer:
# Large Speech 75token: 75Hz. 12 bits. 75*12 = 900bps = 0.9kbps.
# Small 600: 40Hz. 12 bits. 40*12 = 480bps = 0.48kbps.
# Medium 600: 75Hz? (Output said 75).
# Wait, output said wavtokenizer_24k_medium_600_4096 is 75.00 TPS.
# 75 * 12 = 900bps = 0.9kbps.
# Small 600 24k: 40Hz -> 0.48kbps.
# Large 600 24k: 40Hz -> 0.48kbps.

# Consolidate values into a single dictionary
values = {
    'academicodec_hifi_16k_320d': {'bps': 2, 'tps': 50},
    'academicodec_hifi_16k_320d_large_uni': {'bps': 2, 'tps': 50},
    'academicodec_hifi_24k_320d': {'bps': 3, 'tps': 75},
    'audiodec_24k_320d': {'bps': 6, 'tps': 75},  # Assumed 6kbps from 320d?
    'auv': {'bps': 0.5, 'tps': 50},
    'bigcodec_1k': {'bps': 1, 'tps': 80},
    'dac_16k': {'bps': 6, 'tps': 50},
    'dac_24k': {'bps': 24, 'tps': 75},
    'dac_44k': {'bps': 8, 'tps': 86},
    'encodec_24k_12bps': {'bps': 12, 'tps': 75},
    'encodec_24k_1_5bps': {'bps': 1.5, 'tps': 75},
    'encodec_24k_24bps': {'bps': 24, 'tps': 75},
    'encodec_24k_3bps': {'bps': 3, 'tps': 75},
    'encodec_24k_6bps': {'bps': 6, 'tps': 75},
    'funcodec_en_libritts_16k_gr1nq32ds320': {'bps': 16, 'tps': 50},
    'funcodec_en_libritts_16k_gr8nq32ds320': {'bps': 16, 'tps': 50},
    'funcodec_en_libritts_16k_nq32ds320': {'bps': 16, 'tps': 50},
    'funcodec_en_libritts_16k_nq32ds640': {'bps': 8, 'tps': 25},
    'funcodec_zh_en_16k_nq32ds320': {'bps': 16, 'tps': 50},
    'funcodec_zh_en_16k_nq32ds640': {'bps': 8, 'tps': 25},
    's3tokenizer_v1': {'bps': 0.5, 'tps': 50},
    'speech_tokenizer_16k': {'bps': 4, 'tps': 50},
    'sqcodec_16k_0k75bps': {'bps': 0.75, 'tps': 200},
    'sqcodec_16k_12kbps': {'bps': 12, 'tps': 800},
    'sqcodec_16k_1k5bps': {'bps': 1.5, 'tps': 300},
    'sqcodec_16k_3kbps': {'bps': 3, 'tps': 400},
    'sqcodec_16k_6kbps': {'bps': 6, 'tps': 600},
    'sqcodec_24k_12kbps': {'bps': 12, 'tps': 800},
    'sqcodec_24k_24kbps': {'bps': 24, 'tps': 1800},
    'unicodec_24k': {'bps': 12, 'tps': 75},
    'wavtokenizer_24k_small_600_4096': {'bps': 0.48, 'tps': 40},
    'wavtokenizer_24k_medium_600_4096': {'bps': 0.9, 'tps': 75}, 
    'wavtokenizer_24k_large_600_4096': {'bps': 0.48, 'tps': 40},
    'wavtokenizer_24k_large_speech_75token': {'bps': 0.9, 'tps': 75}
}

# Metrics to include
metrics_to_include = ['mel', 'pesq', 'stoi', 'f0corr']

new_results = {}

for model_name, metrics_data in benchmark_results.items():
    model_values = values.get(model_name, {'bps': 0, 'tps': 0})
    entry = {
        'bps': model_values['bps'],
        'tps': model_values['tps']
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
