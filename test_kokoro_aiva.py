"""Quick test: Kokoro 'hello Aiva' pronunciation."""
import os, wave, numpy as np, scipy.signal as sps
from kokoro import KPipeline

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testaudio", "voice_samples")
os.makedirs(OUT_DIR, exist_ok=True)

KOKORO_SR = 24000
TARGET_SR = 16000

# Use the American pipeline and try with "hello Aiva"
p = KPipeline(lang_code="a")

for voice in ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael"]:
    out_path = os.path.join(OUT_DIR, f"kokoro_aiva_{voice}.wav")
    try:
        gen = p("hello Aiva", voice=voice, speed=1.0)
        chunks = [audio for _, _, audio in gen]
        if not chunks:
            print(f"FAIL: {voice} (empty)")
            continue
        audio = np.concatenate(chunks).astype(np.float32)
        if KOKORO_SR != TARGET_SR:
            num_s = int(len(audio) * TARGET_SR / KOKORO_SR)
            audio = sps.resample(audio, num_s).astype(np.float32)
        audio_int16 = np.clip(audio * 32767, -32767, 32767).astype(np.int16)
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(TARGET_SR)
            wf.writeframes(audio_int16.tobytes())
        print(f"OK: {out_path}")
    except Exception as e:
        print(f"FAIL: {voice} - {e}")
print("Done")
