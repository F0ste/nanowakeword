"""Generate one 'hello Ava' sample per Kokoro American voice for listening test."""
import os
import wave
import numpy as np
import scipy.signal as sps

try:
    from kokoro import KPipeline
except ImportError:
    print("Please: pip install kokoro>=0.9.4")
    exit(1)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testaudio", "voice_samples")
TARGET_SR = 16000
KOKORO_SR = 24000

# All American voices to test
VOICES = [
    # currently in config
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    # new candidates
    "af_nova", "af_alloy", "af_aoede", "af_river", "af_kore",
    "am_echo", "am_fenrir", "am_liam", "am_onyx", "am_puck",
]

os.makedirs(OUT_DIR, exist_ok=True)

p = KPipeline(lang_code="a")

for voice in VOICES:
    out_path = os.path.join(OUT_DIR, f"{voice}.wav")
    if os.path.exists(out_path):
        print(f"SKIP (exists): {voice}")
        continue
    try:
        gen = p("hello Ava", voice=voice, speed=1.0)
        chunks = [audio for _, _, audio in gen]
        if not chunks:
            print(f"FAIL (empty): {voice}")
            continue
        audio = np.concatenate(chunks).astype(np.float32)
        if KOKORO_SR != TARGET_SR:
            num_s = int(len(audio) * TARGET_SR / KOKORO_SR)
            audio = sps.resample(audio, num_s).astype(np.float32)
        audio_int16 = np.clip(audio * 32767, -32767, 32767).astype(np.int16)
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SR)
            wf.writeframes(audio_int16.tobytes())
        print(f"OK : {voice} -> {out_path}")
    except Exception as e:
        print(f"FAIL: {voice} - {e}")

print("\nDone. Samples in:", OUT_DIR)
