# NanoWakeWord Project

## Conda Environment

**All Python commands must run inside the `nww` conda environment.**

```powershell
conda activate nww
```

Or use `conda run -n nww python <script>` for one-off commands.

## Key Commands

| Task | Command |
|---|---|
| Generate clips | `python nanowakeword/trainer.py -c config_multilang.yaml -G` |
| Full pipeline | `python nanowakeword/trainer.py -c config_multilang.yaml -G -T` |
| Test AAR SDK | `python test_aar_sdk.py <audio.wav> [threshold]` |
| Convert audio to 16kHz mono wav | `ffmpeg -i input.mp3 -acodec pcm_s16le -ar 16000 -ac 1 output.wav` |

## Audio Format

- Standard: **16kHz, 16-bit, mono, .wav**
- Training data must be 16kHz mono. Use `ffmpeg` or `tools/batch_audio_preprocess.py` to convert.

## Project Structure

- `nanowakeword/` — core Python package
- `android-sdk/wakeword/` — Android AAR library
- `config_multilang.yaml` — active training config
- `CONFIGURATION_GUIDE.md` — full config reference
- `test_aar_sdk.py` — desktop test harness for AAR pipeline

## Voice Call System (WIP)

Goal: phone-call-style AI conversation. Independent project, separate from NanoWakeword.

### Architecture

```
Phone Call (Twilio / SIP)
  └─ audio stream → Server

Server
  ├─ STT  → Deepgram / Whisper / custom API (speech → text)
  ├─ LLM  → custom API (text in, text out)
  └─ TTS  → ElevenLabs / Cartesia / custom API (text → speech)
       └─ audio stream → back to phone call

Cycle:
  call connected → robot speaks greeting → user can interrupt anytime
  → user speaks → STT → LLM → TTS → robot responds
  → conversation continues with interruption support
```

Key requirements:
- Robot proactively speaks default greeting on call connect
- User can interrupt robot speech at any time (barge-in)
- Multi-language: Chinese, English, Cantonese
- External API integration for STT/LLM/TTS

### Framework Evaluation (phone-call scenario)

| | LiveKit Agents | Pipecat | Patter |
|---|---|---|---|
| ⭐ | 10.9k | 12.8k | 515 |
| Phone call | Twilio/SIP (via LiveKit) | Twilio/Daily | ✅ Twilio/Telnyx/Plivo (native) |
| Server-side wake word | Custom audio tap | Custom audio tap | Custom audio tap |
| Custom LLM API | ✅ `llm_node()` | ✅ `_process_context()` | ✅ provider swap |
| Interruption | Semantic + VAD (best) | VAD + strategies | VAD + Krisp |
| Complexity to setup | Medium | Medium | Lowest |

### Research Docs

- `voice_call_systems_report.md` — full comparison of 14 open-source projects

---

## Android SDK Tips

### Build & Install

| Task | Command |
|---|---|
| Build AAR | `android-sdk\gradlew.bat :wakeword:assembleRelease` |
| Build APK (testapp) | `android-sdk\gradlew.bat :testapp:assembleDebug` |
| Build APK (example) | `android-sdk\gradlew.bat :app:assembleDebug` |
| Install to device | `adb install -r testapp\build\outputs\apk\debug\testapp-debug.apk` |

Output paths:
- AAR: `android-sdk\wakeword\build\outputs\aar\wakeword-release.aar`
- APK (testapp): `android-sdk\testapp\build\outputs\apk\debug\testapp-debug.apk`

### Model Replacement Workflow

When you retrain the wake word model and need to update the APK:

1. Copy new models to assets:
   ```
   trained_models/<name>/model/<name>.onnx → testapp/src/main/assets/models/wakeword_full.onnx
   trained_models/<name>/model/<name>_lite.onnx → testapp/src/main/assets/models/wakeword_lite.onnx
   ```
2. Keep `melspectrogram.onnx` and `embedding_model.onnx` unchanged (feature extractors).
3. Rebuild & install.

### Feature Frame Alignment (CRITICAL)

The Android `FeatureExtractor` (in the AAR) has a **hardcoded frame count** (bytecode: `bipush 21`).
The ONNX wake word model has a **fixed input shape** `[batch, frames, 96]`.

| clip_length_samples | mel frames | embedding output | matches |
|---|---|---|---|
| 20480 (1.28s) | ~124 | **7 frames** | ❌ too short |
| **32480 (2.03s)** | ~197 | **16 frames** | ✅ matches ACAV100M / RACON |
| 39000 (2.44s) | ~236 | **21 frames** | only old 21-frame model |

**Rule**: `clip_length_samples` MUST produce the same frame count as the Android FeatureExtractor expects.
Formula: `n_frames = (mel_frames - 76) // 8 + 1`, where mel_frames ≈ 97 × duration_seconds.

If model frame count differs from FeatureExtractor's hardcoded value, you MUST either:
- Retrain with matching `clip_length_samples`, OR
- Bytecode-patch `a/a.class` in the AAR (change `bipush 21` → `bipush NN`), repack AAR, rebuild APK

### ADB Wireless Debugging

When USB is unreliable:

1. Ensure device and computer are on same WiFi (use hotspot if needed).
2. Find device IP (from device settings or previous connections).
3. Connect: `adb connect <IP>:<port>`
   - Default ADB port: `5555`
   - Wireless debugging port: random 5-digit number (shown in Developer Options).
4. Install: `adb -s <IP>:<port> install -r <apk>`.
5. Device must have "Wireless debugging" or "ADB over network" enabled in Developer Options.

### YAML Config Pitfalls

- **Duplicate keys**: YAML silently uses the LAST occurrence. Check `clip_length_samples` only appears once.
- **`--overwrite`**: CLI flag ORs with config's `overwrite`, so `--overwrite` on CLI overrides `overwrite: false` in config.
- **`transform_clips / train_model`**: CLI flags (`-t`, `-T`) also OR with config values. Setting `train_model: false` in config but passing `-T` on CLI will still train.

### Training Data Tips

**Generating test audio:**
```python
from kokoro import KPipeline
pipeline = KPipeline(lang_code='a')
gen = pipeline("hello Ava", voice="if_sara", speed=1.0)
for _, _, audio in gen:
    sf.write("test.wav", audio, 24000)
```

**Generating noise clips for negative samples:**
Use numpy to create white/pink/brown noise, low-freq hum, click/burst impulses.
Place in `data/negative/` and re-extract `negative_features.npy`.

**Adding new negative data without full re-extraction:**
Comment out other jobs in `feature_generation_manifest`, leaving only `neg_feature_1`.
Run: `nanowakeword -c config_multilang.yaml -t --overwrite`

### Common Training Crashes

| Error | Cause | Fix |
|---|---|---|
| `stack expects equal size, got [21,96] and [16,96]` | Validation background (RACON/ACAV) has different frame count than training data | Ensure all .npy have same frame count, or set matching `clip_length_samples` |
| Training hangs at 41% | Validation iterating over millions of background samples | Reduce `bv` to smaller dataset (e.g. project's own `negative_features.npy`) |
| `bipush 21` → 0.0000 scores | Android FeatureExtractor window size ≠ model input shape | Bytecode-patch the AAR or retrain with matching frames |
