"""Generate hello Ava samples using Piper TTS with different speaker IDs."""
import os
import wave
import numpy as np
import scipy.signal as sps
from piper.voice import PiperVoice, SynthesisConfig

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testaudio", "voice_samples")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL = "NwwResourcesModel/tts_models/en_US-libritts_r-medium.onnx"
TARGET_SR = 16000

voice = PiperVoice.load(MODEL)
print(f"Model loaded: {voice.config.sample_rate}Hz, speakers={voice.config.num_speakers}")

# Try 10 different speaker IDs
for sid in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    out_path = os.path.join(OUT_DIR, f"piper_spk{sid}.wav")
    if os.path.exists(out_path):
        print(f"SKIP: {out_path}")
        continue
    try:
        cfg = SynthesisConfig(speaker_id=sid)
        audio_chunks = list(voice.synthesize("hello Ava", cfg))
        audio = b"".join(c.audio_int16_bytes for c in audio_chunks)
        audio_arr = np.frombuffer(audio, dtype=np.int16)

        src_sr = voice.config.sample_rate
        if src_sr != TARGET_SR:
            num_s = int(len(audio_arr) * TARGET_SR / src_sr)
            audio_arr = sps.resample(audio_arr.astype(np.float32), num_s).astype(np.int16)

        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SR)
            wf.writeframes(audio_arr.tobytes())
        print(f"OK: piper_spk{sid}.wav")
    except Exception as e:
        print(f"FAIL: piper_spk{sid} - {e}")

print("Done.")
