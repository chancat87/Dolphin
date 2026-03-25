import copy
from pathlib import Path
from typing import List, Dict, Union, Optional

import torch
import torchaudio
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch_complex.tensor import ComplexTensor

from dolphin.model import DefaultFrontend

frontend = None


def extract_feats(audios: List[Union[Path, torch.Tensor]], configs: Dict) -> Dict[str, torch.Tensor]:
    feats = []
    for audio in audios:
        if isinstance(audio, torch.Tensor):
            waveform = audio
            sample_rate = 16000
        else:
            waveform, sample_rate = torchaudio.load(audio)

        # single channel
        channel_nums = waveform.size(0)
        if channel_nums != 1:
            waveform = waveform[0, :].unsqueeze(0)

        # sample rate 16k
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # fbank feats
        if "frontend_conf" in configs["dataset_conf"]:
            global frontend
            if frontend is None:
                frontend = DefaultFrontend(**configs["dataset_conf"]["frontend_conf"])
                frontend.eval()
            lengths = torch.tensor([waveform.size(-1)], dtype=torch.long)
            feats.append(frontend(waveform, lengths)[0][0])
        else:
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