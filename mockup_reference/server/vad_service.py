from __future__ import annotations
import io
import logging
import threading
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)
_SAMPLE_RATE = 16_000 #sileros required sample rate
_WINDOW_SIZE = 512 #sample size
_SPEECH_THRESHOLD = 0.5
_MIN_SPEECH_WINDOWS = 2 #require at least 2 consecutive windows to confirm speech
_model = None
_utils = None
_model_lock = threading.Lock()


def load_model():
    """Load silero vad from torch once and cache it"""
    global _model, _utils

    if _model is not None:
        return _model, _utils

    with _model_lock:
        if _model is not None:
            return _model, _utils

        import torch

        logger.info("Loading silero model from torch")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        model.eval()
        _model = model
        _utils = utils
        logger.info("silero model loaded")

    return _model, _utils


def _convert_webm_to_pcm(audio_bytes: bytes):
    """convert audio bytes to float array, returns an array of size n with float values between -1.0 and 1.0"""
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise RuntimeError("pydub is required for audio conversion") from e

    try:
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
    except Exception as e:
        raise ValueError(f"Failed to decode audio bytes: {e}") from e

    # resample according to silero's requirements
    segment = segment.set_frame_rate(_SAMPLE_RATE).set_channels(1).set_sample_width(2)

    # convert to int array
    raw = np.frombuffer(segment.raw_data, dtype=np.int16).astype(np.float32)
    raw /= 32768.0  # bring it in range of -1 and 1

    if raw.size == 0:
        raise ValueError("Decoded audio is empty")

    return raw


def contains_speech(audio_bytes: bytes) -> Tuple[bool, float]:
    """determines whether audio has human voice or not, this is a blocking function"""
    import torch  #heavy import hence inside fn
    model, _ = load_model()
    pcm = _convert_webm_to_pcm(audio_bytes)

    #silero wants a float32 array
    audio_tensor = torch.from_numpy(pcm)
    speech_window_count = 0
    max_prob = 0.0
    total_windows = 0

    model.reset_states()

    # visit full audio in sliding window type approach
    for start in range(0, len(pcm) - _WINDOW_SIZE + 1, _WINDOW_SIZE):
        window = audio_tensor[start : start + _WINDOW_SIZE]
        with torch.no_grad():
            prob: float = model(window, _SAMPLE_RATE).item()

        max_prob = max(max_prob, prob)
        total_windows += 1
        if prob >= _SPEECH_THRESHOLD:
            speech_window_count += 1

    has_speech = speech_window_count >= _MIN_SPEECH_WINDOWS
    logger.debug("windows=%d speech_windows=%d  max_prob=%.3f has_speech=%s", total_windows, speech_window_count, max_prob, has_speech)

    return has_speech, max_prob