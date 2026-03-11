# encoding: utf8

import logging
import warnings

LOGGING_FORMAT="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
warnings.filterwarnings("ignore", category=FutureWarning)
# filter framework interanl logs
logging.getLogger("espnet").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("dolphin").setLevel(logging.INFO)

import yaml
import tqdm
import pydub
import logging
import hashlib
import os.path
import argparse
import dataclasses
import numpy as np
from pathlib import Path
from argparse import Namespace
from os.path import dirname, join, abspath, join
from distutils.util import strtobool
from typing import Union, Optional, Tuple, List, Dict, Any

import torch
import modelscope
from modelscope.models.audio.funasr.model import GenericFunASR
try:
    import torch_npu
    torch_npu_is_imported = True
except:
    torch_npu_is_imported = False

from dolphin.audio import load_audio, convert_audio
# from dolphin.model import DolphinSpeech2Text, TranscribeResult, TranscribeSegmentResult
from dolphin.processor import extract_feats
from dolphin.model import (ASRModel, init_speech_model,
                           TranscribeResult, TranscribeSegmentResult)
from dolphin.languages import LANGUAGE_REGION_CODES, LANGUAGE_CODES
from dolphin.constants import SPEECH_LENGTH
from dolphin.model_registry import MODELS
from dolphin.tokenizer import init_tokenizer, BaseTokenizer

logger = logging.getLogger("dolphin")

VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"



def str2bool(value: str) -> bool:

    return bool(strtobool(value))


def parser_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str, help="audio file path")
    parser.add_argument("--model", type=str, default="small", help="model name (default: small)")
    parser.add_argument("--model_dir", type=Path, default=None, help="model checkpoint download diretory")
    parser.add_argument("--lang_sym", type=str, default=None, help="language symbol (e.g. zh)")
    parser.add_argument("--region_sym", type=str, default=None, help="regiion symbol (e.g. CN)")
    parser.add_argument("--device", type=str, default=None, help="torch device (default: None)")
    parser.add_argument("--normalize_length", type=str2bool, default=False, help="whether to normalize length (default: false)")
    parser.add_argument("--padding_speech", type=str2bool, default=False, help="whether padding speech to 30 seconds (default: false)")
    parser.add_argument("--predict_time", type=str2bool, default=True, help="whether predict timestamp (default: true)")
    parser.add_argument("--beam_size", type=int, default=5, help="number of beams in beam search (default: 5)")
    parser.add_argument("--maxlenratio", type=float, default=0.0, help="Input length ratio to obtain max output length (default: 0.0)")

    args = parser.parse_args()
    return args


def detect_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch_npu_is_imported and torch_npu.npu.is_available():
        device = "npu"
    else:
        device = "cpu"

    return device


def seconds_to_hms(total_seconds: int):
    total_seconds = int(total_seconds)
    hours = total_seconds // 3600
    remaining_seconds = total_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def load_model(
    model_name: str,
    model_dir: Union[Path, str],
    device: Optional[Union[str, torch.device]] = None,
) -> ASRModel:
    """
    Load model.

    Args:
        model_name: model name (e.g. small)
        model_dir: model download directory
        device: the accelerator device

    Returns:
        ASRModel instance
    """
    if device is None:
        device = detect_device()
        logger.info(f"auto detect device: {device}")

    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)

    download_model = True
    model_ckpt_file = model_dir / f"{model_name}.pt"
    if model_ckpt_file.exists():
        with open(model_ckpt_file, "rb") as f:
            model_bytes = f.read()

        if hashlib.sha256(model_bytes).hexdigest() == MODELS[model_name]["sha256"]:
            download_model = False
        else:
            logger.warning("model SHA256 chechsum mismatch, redownload model...")

    if download_model:
        model_dir.mkdir(parents=True, exist_ok=True)
        modelscope.snapshot_download(
            model_id=MODELS[model_name]["model_id"],
            local_dir=model_dir,
            allow_file_pattern=None,
            repo_type="model",
        )

    assert (model_dir / "config.yaml").exists(), "model config not found, please redownload model!"
    with open(model_dir / "config.yaml") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    model = init_speech_model(configs)
    state_dict = torch.load(model_dir / f"{model_name}.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    model.model_configs = configs
    model.device = device

    return model


def validate_lang_region(lang_sym: str, region_sym: str):

    if all([lang_sym, region_sym]):
        if f"{lang_sym}-{region_sym}" not in LANGUAGE_REGION_CODES:
            raise Exception("Unsupport language or region!")
    elif any([lang_sym, region_sym]):
        if lang_sym is not None and region_sym is None:
            assert lang_sym in LANGUAGE_CODES, "Unsupport language!"
        elif lang_sym is None and region_sym is not None:
            assert False, "If you specify a dialect, you must configure the language!"

    return True


def transcribe_long(
        model: ASRModel,
        audio: str,
        lang_sym: str = None,
        region_sym: str = None,
        predict_time: bool = True,
        padding_speech: bool = False,
        decoding_method: str = "attention_rescoring",
        beam_size: int = 10,
        **kwargs,
    ) -> List[TranscribeSegmentResult]:
    """
    Transcribe audio to text.

    Args:
        model: model instance
        audio: audio path
        lang_sym: language symbol (e.g. zh)
        region_sym: region symbol (e.g. CN)
        predict_time: whether predict timestamp (default: true)
        padding_speech: depreacted, whether padding speech to 30 seconds (default: false)
        decoding_method: decoding methods, supports: attetion_rescoring (default), attention

    Returns:
        List[TranscribeSegmentResult]
    """
    results = []

    validate_lang_region(lang_sym, region_sym)

    logging.info("download vad model")
    vad_model_dir = Path(os.path.expanduser("~/.cache/dolphin/speech_fsmn_vad"))
    vad_model_dir.mkdir(parents=True, exist_ok=True)
    if not (vad_model_dir / "model.pt").exists():
        modelscope.snapshot_download(
            model_id=VAD_MODEL,
            local_dir=vad_model_dir,
            allow_file_pattern=None,
            repo_type="model",
        )

    logger.info("loading vad model")
    vad_model = GenericFunASR(
        vad_model_dir,
        max_single_segment_time=SPEECH_LENGTH*1000,
        device="cpu",
        disable_update=True,
    )

    # convert audio to sample rate 16k Mono channel audio
    tmp_audio = f"{audio}.wav"
    convert_audio(audio, tmp_audio)

    logger.info("run vad model")
    segments = vad_model(input=tmp_audio, disable_pbar=True)[0]["value"]

    logger.info("decoding...")
    tokenizer = init_tokenizer(model.model_configs)

    audio_segment = pydub.AudioSegment.from_wav(tmp_audio)
    for seg in segments:
        s, e = seg
        raw_data = audio_segment[s:e].set_channels(1).raw_data
        waveform: torch.Tensor = torch.frombuffer(raw_data, dtype=torch.int16).to(torch.float32) / 32768.0
        batch = extract_feats([waveform.unsqueeze(0)], model.model_configs)
        batch["feats"] = batch["feats"].to(model.device)
        batch["feats_lengths"] = batch["feats_lengths"].to(model.device)

        ret = model.decode(
            methods=[decoding_method],
            speech=batch["feats"],
            speech_lengths=batch["feats_lengths"],
            beam_size=beam_size,
            infos={"tokenizer": tokenizer}
        )
        tokens = ret[decoding_method][0].tokens
        nonspecial_tokens = _filter_nonspecial_tokens(tokens, tokenizer)
        lang = tokenizer.ids2tokens([tokens[0]])[0][1:-1]
        region = tokenizer.ids2tokens([tokens[1]])[0][1:-1]
        result = TranscribeResult(
            text=tokenizer.detokenize(tokens)[0],
            text_nospecial=tokenizer.detokenize(nonspecial_tokens)[0],
            language=lang,
            region=region,
        )

        st = seconds_to_hms(s/1000)
        et = seconds_to_hms(e/1000)
        logger.info(f"segment: {st} - {et}, lang: {result.language}, region: {result.region}, text: {result.text_nospecial}")
        result_json = dataclasses.asdict(result)
        result_json.update({
            "start": round(s/1000, 2),
            "end": round(e/1000, 2)
        })
        segment_result = TranscribeSegmentResult(**result_json)
        results.append(segment_result)

    # clean tmp audio file
    Path(tmp_audio).unlink(missing_ok=True)

    return results


def _filter_nonspecial_tokens(tokens: List[int], tokenizer: BaseTokenizer) -> List[int]:
    """
    Filter out special tokens from the token list.

    Args:
        tokens: list of token ids
        tokenizer: tokenizer instance

    Returns:
        list of non-special token ids
    """
    last_time_id = tokenizer.tokens2ids(["<30.00>"])[0]
    nonspecial_tokens = list(filter(lambda x: x > last_time_id , tokens))

    return nonspecial_tokens


def transcribe(
        model: ASRModel,
        audio: str,
        lang_sym: str = None,
        region_sym: str = None,
        predict_time: bool = True,
        padding_speech: bool = False,
        decoding_method: str = "attention_rescoring",
        beam_size: int = 10,
        **kwargs,
    ) -> TranscribeResult:
    """
    Transcribe audio to text.

    Args:
        model: model instance
        audio: audio path
        lang_sym: language symbol (e.g. zh)
        region_sym: region symbol (e.g. CN)
        predict_time: whether predict timestamp (default: true)
        padding_speech: depreacted, whether padding speech to 30 seconds (default: false)
        decoding_method: decoding methods, supports: attetion_rescoring (default), attention

    Returns:
        TranscribeResult
    """
    validate_lang_region(lang_sym, region_sym)

    logger.info("decoding...")
    batch = extract_feats([audio], model.model_configs)
    batch["feats"] = batch["feats"].to(model.device)
    batch["feats_lengths"] = batch["feats_lengths"].to(model.device)

    tokenizer = init_tokenizer(model.model_configs)
    ret = model.decode(
        methods=[decoding_method],
        speech=batch["feats"],
        speech_lengths=batch["feats_lengths"],
        beam_size=beam_size,
        infos={"tokenizer": tokenizer}
    )

    tokens = ret[decoding_method][0].tokens
    nonspecial_tokens = _filter_nonspecial_tokens(tokens, tokenizer)
    lang = tokenizer.ids2tokens([tokens[0]])[0][1:-1]
    region = tokenizer.ids2tokens([tokens[1]])[0][1:-1]

    result = TranscribeResult(
        text=tokenizer.detokenize(tokens)[0],
        text_nospecial=tokenizer.detokenize(nonspecial_tokens)[0],
        language=lang,
        region=region,
    )

    logger.info(f"decode result, language: {result.language}, region: {result.region}, text: {result.text}")
    return result


def cli():
    args = parser_args()

    model = args.model
    if model not in MODELS:
        logging.error(f"Unknown model {model}, Dolphin open source {tuple(MODELS.keys())} model, please config the correct model.")
        return

    model_dir = args.model_dir if args.model_dir else os.path.expanduser(f"~/.cache/dolphin/{model}")
    model_dir = Path(model_dir)

    logger.info("loading asr model")
    device = args.device if args.device else detect_device()
    model_instance = load_model(model, model_dir, device)

    audio_duration = pydub.AudioSegment.from_file(args.audio).duration_seconds
    transcribe_fn = transcribe_long if audio_duration > SPEECH_LENGTH else transcribe
    transcribe_params = {
        "model": model_instance,
        "audio": args.audio,
        "lang_sym": args.lang_sym,
        "region_sym": args.region_sym,
        "predict_time": args.predict_time,
        "padding_speech": args.padding_speech,
    }
    transcribe_fn(**transcribe_params)


if __name__ == "__main__":
    cli()
