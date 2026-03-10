from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence


def extract_feats(audios: List[Path], configs: Dict) -> Dict[str, torch.Tensor]:
    feats = []
    for audio in audios:
        waveform, sample_rate = torchaudio.load(audio)

        # single channel
        channel_nums = waveform.size(0)
        if channel_nums != 1:
            waveform = waveform[0, :].unsqueeze(0)

        # sample rate 16k
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # fbank feats
        waveform = waveform * (1 << 15)
        mel = torchaudio.compliance.kaldi.fbank(waveform=waveform, **configs["dataset_conf"]["fbank_conf"])
        feats.append(mel)

    feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
    feats_lengths = torch.tensor([feat.size(0) for feat in feats], dtype=torch.long)

    batch = {
        "feats": feats,
        "feats_lengths": feats_lengths,
    }

    return batch
