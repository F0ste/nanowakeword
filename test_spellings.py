"""Test different spellings of Ava to force EY-va pronunciation."""
import os, wave, numpy as np, scipy.signal as sps
from piper.voice import PiperVoice, SynthesisConfig

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testaudio", "voice_samples")
os.makedirs(OUT_DIR, exist_ok=True)
MODEL = "NwwResourcesModel/tts_models/en_US-libritts_r-medium.onnx"
TARGET_SR = 16000

voice = PiperVoice.load(MODEL)
print(f"Speakers: {voice.config.num_speakers}")

spellings = ["hello Ava", "hello Ayva", "hello Aiva", "hello Eyva", "hello Aeyva"]

for spelling in spellings:
    out_path = os.path.join(OUT_DIR, f"spell_{spelling.replace(' ','_')}.wav")
    try:
        cfg = SynthesisConfig(speaker_id=0)
        chunks = list(voice.synthesize(spelling, cfg))
        audio = b"".join(c.audio_int16_bytes for c in chunks)
        arr = np.frombuffer(audio, dtype=np.int16)
        if voice.config.sample_rate != TARGET_SR:
            num_s = int(len(arr) * TARGET_SR / voice.config.sample_rate)
            arr = sps.resample(arr.astype(np.float32), num_s).astype(np.int16)
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_SR)
            wf.writeframes(arr.tobytes())
        print(f"OK: {out_path}")
    except Exception as e:
        print(f"FAIL: {spelling} - {e}")
print("Done")
