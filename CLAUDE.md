# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dolphin is a multilingual, multitask ASR (Automatic Speech Recognition) model developed by DataoceanAI and Tsinghua University. It supports 40 Eastern languages and 22 Chinese dialects, performs speech recognition, voice activity detection (VAD), segmentation, and language identification (LID).

## Architecture

Dolphin follows the Whisper/OWSM design with a joint CTC-Attention architecture:
- **Encoder**: E-Branchformer-based
- **Decoder**: Standard Transformer
- **Two-level language token system**: `<lang>` (e.g., `<zh>`) + `<region>` (e.g., `<CN>`)

Key modules:
- `transcribe.py` - Main entry point: CLI parsing, model loading, transcription orchestration
- `model.py` - ASR model architecture (Stft, encoder, decoder), `TranscribeResult` dataclasses
- `search.py` - Decoding algorithms: CTC greedy search, CTC prefix beam search, attention beam search, attention rescoring
- `processor.py` - Feature extraction (fbank features)
- `tokenizer.py` - Tokenization (CharTokenizer, BpeTokenizer)
- `audio.py` - Audio loading using FFmpeg subprocess
- `model_registry.py` - Model registry with SHA256 verification

## Commands

### Installation
```bash
pip install -U dataoceanai-dolphin
# or from source:
pip install git+https://github.com/SpeechOceanTech/Dolphin.git
```

### CLI Usage
```bash
dolphin audio.wav
dolphin audio.wav --model small --model_dir /data/models/dolphin/
dolphin audio.wav --model small --lang_sym "zh" --region_sym "CN"
```

### Python API
```python
import dolphin
waveform = dolphin.load_audio("audio.wav")
model = dolphin.load_model("small", "/data/models/dolphin", "cuda")
result = model(waveform, lang_sym="zh", region_sym="CN")
print(result.text)
```


### Package Structure
```
dolphin/
  __init__.py      - Public API exports (load_audio, load_model, transcribe)
  __main__.py      - CLI entry point (python -m dolphin)
  transcribe.py    - Main transcription logic, CLI, model loading
  model.py         - ASR model, encoder/decoder, result dataclasses
  search.py        - Beam search and rescoring decoding methods
  processor.py     - Feature extraction (fbank features)
  tokenizer.py     - Tokenization (char/bpe)
  audio.py         - Audio loading via FFmpeg subprocess
  model_registry.py - Model definitions with SHA256 hashes
  mask.py          - Attention masking utilities
  common.py        - Padding, CMVN normalization utilities
  languages.py     - Language/region code mappings
  constants.py     - Sample rate (16k), speech length (30s) constants
  assets/          - BPE model, config, stats files
```

## Model Registry

Models are downloaded from DataoceanAI's ModelScope hub. Each model has a SHA256 hash for integrity verification (`model_registry.py`). Supported models: `base`, `small`, `small.zh`, `small.zh.streaming`.

## Device Support

Supports CUDA, MPS (Apple Silicon), NPU (Huawei Ascend), and CPU. Device detection is automatic via `detect_device()` in `transcribe.py`.
