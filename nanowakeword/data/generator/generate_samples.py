# NanoWakeWord
# Copyright 2025 Arcosoph
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Project Repository: https://github.com/arcosoph/nanowakeword
#
# This software is provided "AS IS", without warranties or conditions of any kind.
# See the License for the specific language governing permissions and limitations.

import logging
import os
import sys
import wave
import traceback
import random
import io
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import scipy.signal as sps
from typing import List, Optional, Union, Dict, Any

# Load .env file for API keys
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

# Module-level setup 
_LOGGER = logging.getLogger(__name__)
_LOADED_VOICE_CACHE: Dict[str, Dict] = {}
DEFAULT_MODELS_DIR = None

# Safe import of utilities and project root initialization 
try:
    from nanowakeword import PROJECT_ROOT
    from nanowakeword.utils.download_files import download_file
    # DEFAULT_MODELS_DIR = PROJECT_ROOT / "resources" / "tts_models"

    DEFAULT_MODELS_DIR = os.path.join("NwwResourcesModel", "tts_models")
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    # DEFAULT_MODELS_DIR = PROJECT_ROOT / "resources" / "tts_models"
    DEFAULT_MODELS_DIR = os.path.join("NwwResourcesModel", "tts_models")
    def download_file(url: str, target_directory: str):
        import requests
        filename = url.split('/')[-1]
        filepath = os.path.join(target_directory, filename)
        _LOGGER.info(f"Downloading {url} to {filepath}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            _LOGGER.info("Download complete.")
            return str(filepath)
        except requests.exceptions.RequestException as e:
            _LOGGER.error(f"Failed to download {url}: {e}")
            return None

# Force onnxruntime (used by Piper TTS) to prefer GPU if available
try:
    import onnxruntime as _ort
    _available = _ort.get_available_providers()
    _gpu_providers = [p for p in _available if p != 'CPUExecutionProvider']
    if _gpu_providers:
        _orig_inference_session = _ort.InferenceSession

        def _gpu_inference_session(path_or_bytes, sess_options=None,
                                   providers=None, provider_options=None, **kwargs):
            if providers is None or providers == ['CPUExecutionProvider']:
                providers = _gpu_providers + ['CPUExecutionProvider']
                provider_options = [{} for _ in providers]
            return _orig_inference_session(path_or_bytes, sess_options=sess_options,
                                           providers=providers,
                                           provider_options=provider_options, **kwargs)

        _ort.InferenceSession = _gpu_inference_session
except Exception:
    pass

_PIPER_AVAILABLE = False
_PiperVoice = _SynthesisConfig = None
try:
    from piper.voice import PiperVoice as _PiperVoice, SynthesisConfig as _SynthesisConfig
    _PIPER_AVAILABLE = True
except ImportError:
    pass

_KOKORO_AVAILABLE = False
_KPipeline = None
try:
    from kokoro import KPipeline as _KPipeline
    _KOKORO_AVAILABLE = True
except ImportError:
    pass

_KOKORO_PIPELINE_CACHE: Dict[str, Any] = {}

# Mapping from lang_code to default voice (core Kokoro v0.9.4+)
# For languages not listed (e.g. 'k'=Korean from wrappers), user must specify a voice.
_KOKORO_DEFAULT_VOICES = {
    'a': 'af_heart',
    'b': 'bf_emma',
    'e': 'ef_dora',
    'f': 'ff_siwis',
    'h': 'hf_alpha',
    'i': 'if_sara',
    'j': 'jf_alpha',
    'p': 'pf_dora',
    'z': 'zf_xiaoxiao',
}

_KOKORO_SAMPLE_RATE = 24000

# --- DashScope / Qwen3-TTS API ---
_DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

_ALIYUN_LANG_MAP = {
    "yue":        ("chinese", "Kiki"),
    "cantonese":  ("chinese", "Kiki"),
    "ko":         ("korean", "Sohee"),
    "korean":     ("korean", "Sohee"),
    "zh":         ("chinese", "Cherry"),
    "chinese":    ("chinese", "Cherry"),
    "en":         ("english", "Cherry"),
    "english":    ("english", "Cherry"),
    "ja":         ("japanese", "Cherry"),
    "japanese":   ("japanese", "Cherry"),
}

_ALIYUN_MODEL = "qwen3-tts-flash"


def _load_voice_from_path(onnx_path: Path) -> Optional[Dict[str, Any]]:
    """Internal helper to load a single voice from a given path and cache it."""
    global _LOADED_VOICE_CACHE
    abs_path_str = str(onnx_path.resolve())

    if abs_path_str in _LOADED_VOICE_CACHE:
        return _LOADED_VOICE_CACHE[abs_path_str]

    json_path = onnx_path.with_suffix(".onnx.json")
    if not (onnx_path.exists() and json_path.exists()):
        _LOGGER.warning(f"Model files not found for '{onnx_path.stem}'. Skipping.")
        return None

    try:
        _LOGGER.info(f"Loading TTS model: {onnx_path.name}...")
        voice_obj = _PiperVoice.load(onnx_path)
        voice_data = {"voice": voice_obj, "path": onnx_path}
        
        num_spk_str = f" | Speakers: {voice_obj.config.num_speakers}" if voice_obj.config.num_speakers else ""
        _LOGGER.info(f"Model loaded successfully{num_spk_str}")
        
        _LOADED_VOICE_CACHE[abs_path_str] = voice_data
        return voice_data
    except Exception as e:
        _LOGGER.error(f"Failed to load '{onnx_path.name}': {e}\n{traceback.format_exc()}")
        return None


def load_voices(
    models: Optional[Union[str, List[str], Dict[str, str]]] = None,
    models_dir: str = str(DEFAULT_MODELS_DIR)
) -> List[Dict]:
    """
    Loads one or more Piper TTS voice models with maximum flexibility.

    Args:
        models (Optional): Can be one of the following:
            - None: Auto-load all models from `models_dir`. If empty, downloads the default.
            - str: A single model name, file path, or a directory path.
            - List[str]: A list of model names or file paths.
            - Dict[str, str]: A dictionary with 'onnx' and 'json' URLs to download.
        models_dir (str): The directory to download models to or load from.

    Returns:
        List[Dict]: A list of loaded voice data dictionaries.
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Determine all model paths that need to be loaded.
    model_paths_to_process = set()

    if models is None:
        # If no specific models are requested, check the default directory.
        existing_models = list(Path(models_dir).glob("*.onnx"))
        if existing_models:
            model_paths_to_process.update(existing_models)
        else:
            # If the directory is empty, download the default model.
            _LOGGER.info("No local models found. Downloading the default multi-speaker model...")
            default_model = "en_US-libritts_r-medium"
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/"
            onnx_url = f"{base_url}{default_model}.onnx"
            json_url = f"{base_url}{default_model}.onnx.json"

            # Download both files. The function should handle failures internally.
            download_file(onnx_url, models_dir)
            download_file(json_url, models_dir)
            
            # After downloading, re-scan the directory to find the new model path.
            model_paths_to_process.update(Path(models_dir).glob(f"{default_model}.onnx"))

    elif isinstance(models, str):
        path = Path(models)
        if path.is_dir():
            model_paths_to_process.update(path.glob("*.onnx"))
        elif path.is_file() and path.suffix == ".onnx":
            model_paths_to_process.add(path)
        else:
            # Assumes it's a model name without extension.
            model_paths_to_process.add(Path(models_dir) / f"{models}.onnx")

    elif isinstance(models, list):
        for item in models:
            path = Path(item)
            if path.is_file() and path.suffix == ".onnx":
                model_paths_to_process.add(path)
            else:
                model_paths_to_process.add(Path(models_dir) / f"{item}.onnx")

    elif isinstance(models, dict) and 'onnx' in models and 'json' in models:
        onnx_path_str = download_file(models['onnx'], models_dir)
        download_file(models['json'], models_dir)
        if onnx_path_str:
            model_paths_to_process.add(Path(onnx_path_str))
    
    # Step 2: Load the voices from the collected paths.
    if not model_paths_to_process:
        _LOGGER.error("No valid TTS models found or downloaded. Cannot generate audio.")
        return []

    _LOGGER.info(f"Found {len(model_paths_to_process)} TTS model(s) to load.")
    
    loaded_voices = [_load_voice_from_path(p) for p in model_paths_to_process]
    
    final_voices = [v for v in loaded_voices if v is not None]
    
    if not final_voices:
         _LOGGER.error("Failed to load any of the found TTS models.")
         
    return final_voices


def _get_kokoro_voices(lang_code: str, voice: Optional[Union[str, List[str]]] = None) -> List[str]:
    """Resolve voice names for a Kokoro language pipeline."""
    if voice is None:
        default = _KOKORO_DEFAULT_VOICES.get(lang_code)
        return [default] if default else []
    if isinstance(voice, str):
        return [voice]
    return voice


def _kokoro_synthesize(
    text: str,
    pipeline,
    voice: str,
    speed: float,
    target_sr: int
) -> Optional[np.ndarray]:
    """Synthesize text with Kokoro, concatenate chunks, resample to target SR."""
    try:
        generator = pipeline(text, voice=voice, speed=speed)
        chunks = [audio for _, _, audio in generator]
        if not chunks:
            return None
        audio = np.concatenate(chunks).astype(np.float32)
        if _KOKORO_SAMPLE_RATE != target_sr:
            num_samples = int(len(audio) * target_sr / _KOKORO_SAMPLE_RATE)
            audio = sps.resample(audio, num_samples).astype(np.float32)
        return audio
    except Exception as e:
        _LOGGER.error(f"Kokoro synthesis failed for '{text[:50]}...': {e}")
        return None


def _save_text_alongside(output_dir: str, wav_filename: str, text: str):
    """Save the synthesized text as a .txt file alongside the .wav file."""
    txt_filename = wav_filename.replace(".wav", ".txt")
    txt_path = os.path.join(output_dir, txt_filename)
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        _LOGGER.warning(f"Failed to save text file for '{wav_filename}': {e}")


def generate_samples(
    text: Union[str, List[str]],
    output_dir: str,
    max_samples: int,
    file_prefix: str = "sample",
    models: Optional[Union[str, List[str], Dict[str, str]]] = None,
    models_dir: str = str(DEFAULT_MODELS_DIR),
    speaker_ids: Optional[Union[int, List[int]]] = None,
    length_scales: List[float] = [1.0],
    noise_scales: List[float] = [0.667],
    noise_w_scales: List[float] = [0.8],
    engine: str = "piper",
    lang_code: str = "a",
    voice: Optional[Union[str, List[str]]] = None,
    speed: float = 1.0
    ):
    """
    Generate audio files from text using Piper or Kokoro TTS engines.

    This function takes one or more text prompts and converts them into
    spoken audio, saving each result as a WAV file. You can control
    the voice, speaker, speech speed, and audio variation for each sample.

    Parameters
    ----------
    text : str or list of str
        The text(s) you want to turn into audio.
    output_dir : str
        The folder where the audio files will be saved.
    max_samples : int
        Total number of audio clips to generate.
    file_prefix : str, optional
        The beginning of the filename for each audio clip (default "sample").
    engine : str, optional
        TTS engine to use: "piper" (default), "kokoro", or "aliyun_tts".

    Piper-specific params:
        models, models_dir, speaker_ids, length_scales, noise_scales, noise_w_scales

    Kokoro-specific params:
        lang_code : str, default "a" (see Kokoro LANG_CODES)
        voice : str or list of str, optional
        speed : float, default 1.0

    Aliyun-TTS-specific params:
        lang_code : str, e.g. "yue" (Cantonese), "ko" (Korean), "zh", "en", etc.
        voice : str or list of str, optional.
            Cantonese: "Kiki" (f), "Rocky" (m)
            Korean: "Sohee" (f)
            Others default to "Cherry"
        speed : float (currently unused by API)
        Requires DASHSCOPE_API_KEY in .env or environment.

    Notes
    -----
    - The audio will be resampled to 16kHz.
    - File names include a timestamp and random number to avoid overwriting.
    - Works with single or multiple text prompts and multiple voice models.
    """

    import time

    os.makedirs(output_dir, exist_ok=True)
    if isinstance(text, str): text = [text]

    text_prompts = (text * ((max_samples // len(text)) + 1))[:max_samples]
    TARGET_SAMPLE_RATE = 16000

    if engine == "kokoro":
        _generate_kokoro(text_prompts, output_dir, max_samples, file_prefix,
                         lang_code, voice, speed, TARGET_SAMPLE_RATE)
        return

    if engine == "aliyun_tts":
        _generate_aliyun_tts(text_prompts, output_dir, max_samples, file_prefix,
                             lang_code, voice, speed, TARGET_SAMPLE_RATE)
        return

    # --- Piper path (default) ---
    if not _PIPER_AVAILABLE:
        _LOGGER.critical("piper-tts is not installed. Please run: pip install piper-tts")
        return

    import itertools

    voices_data = load_voices(models, models_dir)
    if not voices_data:
        _LOGGER.error("Audio generation failed: No TTS models could be loaded.")
        return

    _LOGGER.info(f"Generating {max_samples} samples using {len(voices_data)} loaded voice model(s)...")

    settings_product = itertools.product(
        voices_data,
        length_scales,
        noise_scales,
        noise_w_scales
    )

    settings_iterator = itertools.cycle(settings_product)

    for i, prompt in enumerate(tqdm(text_prompts, desc="Generating Audio", unit="sample")):
        try:
            selected_voice_data, length_scale, noise_scale, noise_w_scale = next(settings_iterator)

            voice_obj, voice_path = selected_voice_data["voice"], selected_voice_data["path"]

            num_speakers = voice_obj.config.num_speakers or 0
            current_speaker_id = None
            if num_speakers > 1:
                if isinstance(speaker_ids, int): current_speaker_id = speaker_ids
                elif isinstance(speaker_ids, list) and speaker_ids: current_speaker_id = random.choice(speaker_ids)
                else: current_speaker_id = random.randint(0, num_speakers - 1)
                current_speaker_id = min(current_speaker_id, num_speakers - 1)

            synthesis_config = _SynthesisConfig(
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w_scale=noise_w_scale,
                speaker_id=current_speaker_id)

            audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in voice_obj.synthesize(prompt, synthesis_config))
            if not audio_bytes: continue

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            if voice_obj.config.sample_rate != TARGET_SAMPLE_RATE:
                num_s = int(len(audio_array) * TARGET_SAMPLE_RATE / voice_obj.config.sample_rate)
                audio_array = sps.resample(audio_array, num_s).astype(np.int16)

            timestamp_ms = int(time.time() * 1000)
            random_num = random.randint(100, 999)
            speaker_tag = f"s{current_speaker_id}" if current_speaker_id is not None else "s0"
            voice_tag = voice_path.stem.split('-')[1]
            out_filename = f"{file_prefix}_{timestamp_ms}_{random_num}_{voice_tag}_{speaker_tag}.wav"

            max_val = np.max(np.abs(audio_array))
            if max_val > 32767:
                audio_array = (audio_array / max_val * 32767)

            audio_float = audio_array.astype(np.float32)

            try:
                kernel_size = 3
                filtered_audio = sps.medfilt(audio_float, kernel_size=kernel_size)
            except Exception as median_error:
                _LOGGER.warning(f"There was a problem applying the Median filter: {median_error}")
                filtered_audio = audio_float

            try:
                sos = sps.butter(4, 7000, 'low', fs=TARGET_SAMPLE_RATE, output='sos')
                final_audio = sps.sosfilt(sos, filtered_audio)
            except Exception as butter_error:
                _LOGGER.warning(f"There was a problem applying the Butterworth filter: {butter_error}")
                final_audio = filtered_audio

            audio_array_clean = np.clip(final_audio, -32767, 32767).astype(np.int16)

            with wave.open(os.path.join(output_dir, out_filename), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_SAMPLE_RATE)
                wf.writeframes(audio_array_clean.tobytes())

            _save_text_alongside(output_dir, out_filename, prompt)

        except Exception as e:
            _LOGGER.error(f"Error on sample {i} ('{prompt}'): {e}\n{traceback.format_exc()}")

    _LOGGER.info("Sample generation complete.")


def _generate_aliyun_tts(
    text_prompts: List[str],
    output_dir: str,
    max_samples: int,
    file_prefix: str,
    lang_code: str,
    voice: Optional[Union[str, List[str]]],
    speed: Union[float, List[float]],
    target_sr: int
):
    """Qwen3-TTS Flash generation via DashScope SDK."""
    if not _DASHSCOPE_API_KEY:
        _LOGGER.critical("DASHSCOPE_API_KEY not set. Add it to .env or environment.")
        return

    try:
        import dashscope
    except ImportError:
        _LOGGER.critical("dashscope is not installed. Please run: pip install dashscope")
        return

    import requests
    import itertools

    lang_type, default_voice = _ALIYUN_LANG_MAP.get(
        lang_code.lower(), ("English", "Cherry")
    )

    if voice is None:
        voices = [default_voice]
    elif isinstance(voice, str):
        voices = [voice]
    else:
        voices = voice

    if isinstance(speed, (int, float)):
        speed_variations = [speed]
    else:
        speed_variations = list(speed)

    setting_cycler = itertools.cycle(itertools.product(voices, speed_variations))

    _LOGGER.info(f"Generating {max_samples} samples via Qwen3-TTS (lang={lang_type}, voices={voices}, speeds={speed_variations})")

    for i, prompt in enumerate(tqdm(text_prompts, desc="Generating Audio (Qwen3-TTS)", unit="sample")):
        try:
            current_voice, current_speed = next(setting_cycler)

            response = dashscope.MultiModalConversation.call(
                model=_ALIYUN_MODEL,
                api_key=_DASHSCOPE_API_KEY,
                text=prompt,
                voice=current_voice,
                language_type=lang_type,
                speech_rate=current_speed,
                stream=False
            )

            if getattr(response, "code", None):
                _LOGGER.error(f"API error on sample {i}: code={response.code}, msg={response.message}")
                continue

            audio_url = None
            output = getattr(response, "output", None)
            if output:
                audio_info = getattr(output, "audio", None)
                if audio_info:
                    audio_url = getattr(audio_info, "url", None)
            if not audio_url:
                _LOGGER.error(f"No audio URL in response for '{prompt[:50]}...'")
                continue

            audio_resp = requests.get(audio_url, timeout=60)
            audio_resp.raise_for_status()

            audio_array = np.frombuffer(audio_resp.content, dtype=np.int16)

            sample_rate = None
            try:
                audio_output = getattr(response.output, "audio", None)
                if audio_output is not None:
                    sample_rate = getattr(audio_output, "sample_rate", None)
            except (KeyError, AttributeError):
                pass
            sample_rate = sample_rate or target_sr
            if sample_rate != target_sr:
                num_s = int(len(audio_array) * target_sr / sample_rate)
                audio_array = sps.resample(audio_array.astype(np.float64), num_s).astype(np.int16)

            timestamp_ms = int(time.time() * 1000)
            random_num = random.randint(100, 999)
            out_filename = f"{file_prefix}_{timestamp_ms}_{random_num}_{current_voice}_s{current_speed:.2f}.wav"

            with wave.open(os.path.join(output_dir, out_filename), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(target_sr)
                wf.writeframes(audio_array.tobytes())

            _save_text_alongside(output_dir, out_filename, prompt)

        except Exception as e:
            _LOGGER.error(f"Error on sample {i} ('{prompt}'): {e}\n{traceback.format_exc()}")

    _LOGGER.info("Sample generation complete (Qwen3-TTS).")


def _generate_kokoro(
    text_prompts: List[str],
    output_dir: str,
    max_samples: int,
    file_prefix: str,
    lang_code: str,
    voice: Optional[Union[str, List[str]]],
    speed: float,
    target_sr: int
):
    """Kokoro TTS generation path."""
    if not _KOKORO_AVAILABLE:
        _LOGGER.critical("kokoro is not installed. Please run: pip install kokoro>=0.9.4")
        return

    import time

    voices = _get_kokoro_voices(lang_code, voice)
    if not voices:
        _LOGGER.error(f"No Kokoro voices available for lang_code='{lang_code}'.")
        return

    if lang_code not in _KOKORO_PIPELINE_CACHE:
        _LOGGER.info(f"Loading Kokoro pipeline for lang_code='{lang_code}'...")
        try:
            _KOKORO_PIPELINE_CACHE[lang_code] = _KPipeline(lang_code=lang_code)
        except Exception as e:
            _LOGGER.error(f"Failed to load Kokoro pipeline: {e}")
            return

    pipeline = _KOKORO_PIPELINE_CACHE[lang_code]

    import itertools
    voice_cycler = itertools.cycle(voices)
    speed_variations = [speed * s for s in [0.95, 1.0, 1.05]]

    _LOGGER.info(f"Generating {max_samples} samples via Kokoro (lang={lang_code}, voices={voices})")

    for i, prompt in enumerate(tqdm(text_prompts, desc="Generating Audio (Kokoro)", unit="sample")):
        try:
            current_voice = next(voice_cycler)
            current_speed = random.choice(speed_variations)

            audio = _kokoro_synthesize(prompt, pipeline, current_voice, current_speed, target_sr)
            if audio is None:
                continue

            audio_int16 = np.clip(audio * 32767, -32767, 32767).astype(np.int16)

            timestamp_ms = int(time.time() * 1000)
            random_num = random.randint(100, 999)
            out_filename = f"{file_prefix}_{timestamp_ms}_{random_num}_{current_voice}.wav"

            with wave.open(os.path.join(output_dir, out_filename), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(target_sr)
                wf.writeframes(audio_int16.tobytes())

            _save_text_alongside(output_dir, out_filename, prompt)

        except Exception as e:
            _LOGGER.error(f"Error on sample {i} ('{prompt}'): {e}\n{traceback.format_exc()}")

    _LOGGER.info("Sample generation complete (Kokoro).")


def main():
    """Main function for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate audio samples using a Piper TTS model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("text", nargs='+', help="One or more text prompts to synthesize.")
    parser.add_argument("--output_dir", required=True, help="Directory to save audio files.")
    parser.add_argument("--max_samples", type=int, default=1, help="Total number of samples to generate.")
    parser.add_argument(
        "--model", 
        default="en_US-libritts_r-medium",
        help="Name of the Piper model or path to a custom .onnx file."
    )
    parser.add_argument(
        "--speaker_id", 
        type=int, 
        default=None,
        help="(For multi-speaker models) Force a specific speaker ID."
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

    generate_samples(
        text=args.text,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        models=args.model,
        speaker_ids=args.speaker_id
    )

if __name__ == "__main__":
    main()


