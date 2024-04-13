import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import os
from argparse import ArgumentParser


emotion_mapping = {
    'angry': 0,
    'disgusted': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'other': 5,
    'sad': 6,
    'surprised': 7,
    'unknown': 8
}


def Emotion_Eval(
    ref_path,
    syn_path,
    device="gpu:0",
    model_type="iic/emotion2vec_base_finetuned",
):
    """
    Args:
        ref_path (str): Path to the reference audio files.
        syn_path (str): Path to the synthesis audio files.
        device (str): Device to run the inference on (default is "gpu:0").
        model_type (str): Which model to use, iic/emotion2vec_base or iic/emotion2vec_base_finetuned.
    """

    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model=model_type,
        model_revision="v2.0.4",
        device=device,
    )

    paths1 = sorted(
        [
            os.path.join(ref_path, filename)
            for filename in os.listdir(ref_path)
            if os.path.isfile(os.path.join(ref_path, filename))
        ]
    )
    paths2 = sorted(
        [
            os.path.join(syn_path, filename)
            for filename in os.listdir(syn_path)
            if os.path.isfile(os.path.join(syn_path, filename))
        ]
    )

    ref_preds = []
    gen_preds = []

    with torch.no_grad():
        res1s = inference_pipeline(paths1)
        res2s = inference_pipeline(paths2)

    for res1, res2 in zip(res1s, res2s):
        score1 = np.array(res1["scores"])
        score2 = np.array(res2["scores"])

        max_index1 = np.argmax(score1)
        max_index2 = np.argmax(score2)

        ref_preds.append(max_index1)
        gen_preds.append(max_index2)

    accuracy_ref = sum(r == p for r, p in zip(ref_preds, gen_preds)) / len(ref_preds)

    labels = []
    for path in paths2:
        emotion_code = path.split("/")[-1].split("-")[0]
        label = emotion_mapping.get(emotion_code, "unknown")
        labels.append(label)
    
    accuracy_groud_truth = sum(r == p for r, p in zip(labels, gen_preds)) / len(ref_preds)

    return accuracy_ref, accuracy_groud_truth


if __name__ == "__main__":
    parser = ArgumentParser(description="Emotion Evaluation")

    parser.add_argument(
        "--ref_path", default="/ref/path", help="path to the reference data"
    )
    parser.add_argument(
        "--syn_path", default="/syn/path", help="path to the generate data"
    )
    parser.add_argument("--device", default="gpu:0", help="device type")
    parser.add_argument(
        "--model_type",
        default="iic/emotion2vec_base_finetuned",
        help="either iic/emotion2vec_base or iic/emotion2vec_base_finetuned",
    )

    args = parser.parse_args()

    accuracy_ref, accuracy_groud_truth = Emotion_Eval(
        args.ref_path, args.syn_path, device=args.device
    )

    print("Acc_ref_audio %2.2f%%"%(accuracy_ref * 100))
    print("Acc_ground_truth %2.2f%%"%(accuracy_groud_truth * 100))
