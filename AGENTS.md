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
