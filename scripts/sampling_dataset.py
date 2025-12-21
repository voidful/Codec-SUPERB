import nlp2
import os
import torch
from tqdm.auto import tqdm

datasets_list = [
    "fluent_speech_commands_test_subset_synth",
    "snips_test_valid_subset_synth",
    "opensinger_synth",
    "m4singer_synth",
    "snips_test_valid_synth",
    "fluent_speech_commands_synth",
    "voxceleb1_synth",
    "quesst14_all_synth",
    "vocalset_synth",
    "Nsynth-test_synth",
    "fsd50k_synth",
    "iemocap_synth",
    "audioset_synth",
    "libri2Mix_test_synth",
    "crema_d_synth",
    "vocal_imitation_synth",
    "libricount_synth",
    "vox_lingua_top10_synth",
    "librispeech_asr_test_synth",
    "gtzan_synth",
    "gtzan_music_speech_synth",
    "beijing_opera_synth",
    "mridangam_synth",
    "gunshot_triangulation_synth",
    "esc50_synth",
    "superb_ks_synth",
    "librispeech_asr_dummy_synth",
    "noisy_vctk_16k_synth",
    "covost2_synth",
    "librispeech_synth",
    "quesst_synth",
    "musdb18_synth",
    "cv_13_zh_tw_synth",
    "beehive_states_synth",
    "ljspeech_synth",
    "opencpop_synth",
    "dcase2016_task2_synth",
    "maestro_synth",
    "vox_lingua_synth"
]
for dataset_name in datasets_list:
    subset_name = dataset_name.replace("_synth", "")
    dataset = load_dataset(f"Codec-SUPERB/{dataset_name}", split='original', streaming=True)
    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)

    save_dir = os.path.join('./sample_100', subset_name)
    os.makedirs(save_dir, exist_o=True)
    txt_path = os.path.join('./sample_100', f"{subset_name}.txt")

    count = 0
    with open(txt_path, 'w') as txt_file:
        for data in shuffled_dataset:
            if count == 100:
                break
            sample_rate = data['audio']['sampling_rate']
            audio_tensor = torch.tensor(data['audio']['array'])
            wav = audio_tensor.unsqueeze(0) if audio_tensor.dim() == 1 else audio_tensor
            file_id = data['id'].replace("/", "_")
            if os.path.splitext(file_id)[1] != '.wav':
                file_id += '.wav'
            path = os.path.join(save_dir, file_id)
            print(path)
            txt_file.write(path.replace("./sample_100/", "") + '\n')
            save_audio(wav, path, sample_rate)
            count += 1
    len(list(nlp2.get_files_from_dir(save_dir)))
