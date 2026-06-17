"""
AAR SDK 等效测试脚本 — 精确模拟 Android FeatureExtractor + CascadeInference 管线
用法: python test_aar_sdk.py <音频文件.wav> [阈值, 默认0.4] [--warmup]
示例:
  python test_aar_sdk.py test.wav                   # 阈值 0.4, 无 warmup
  python test_aar_sdk.py test.wav 0.5               # 阈值 0.5
  python test_aar_sdk.py test.wav 0.4 --warmup      # 启用 21 帧 warmup
"""
import sys
import os
import wave
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("请先安装: pip install onnxruntime"); sys.exit(1)

# ========== 参数 (与 Android FeatureExtractor / CascadeInference 一致) ==========
SR          = 16000
CHUNK       = 1280       # 每帧音频样本 (80ms)
MEL_BINS    = 32         # mel 频带数
FEAT_DIM    = 96         # 特征维度
MEL_WIN     = 76         # embedding 窗口 (mel 帧)
MEL_STRIDE  = 8          # embedding 步长
MEL_EXTRA   = 480        # mel 计算额外音频
MEL_SCALE   = 0.1        # mel 缩放
MEL_BIAS    = 2.0        # mel 偏置
FEAT_FRAMES = 21         # 推理帧数
GATE_DEF    = 0.2        # gate 默认阈值

# 模型路径
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "android-sdk", "wakeword", "src", "main", "assets", "models")
MEL_MODEL  = os.path.join(BASE, "melspectrogram.onnx")
EMB_MODEL  = os.path.join(BASE, "embedding_model.onnx")
LITE_MODEL = os.path.join(BASE, "wakeword_lite.onnx")
FULL_MODEL = os.path.join(BASE, "wakeword_full.onnx")
WAKE_WORD  = "hello Ava"  # 唤醒词文本

# ========== 命令行参数 ==========
if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)

AUDIO_FILE = sys.argv[1]
THRESHOLD  = 0.4
ENABLE_WARMUP = False
for a in sys.argv[2:]:
    if a == "--warmup":
        ENABLE_WARMUP = True
    else:
        try: THRESHOLD = float(a)
        except ValueError: pass

if not os.path.exists(AUDIO_FILE):
    print(f"文件不存在: {AUDIO_FILE}"); sys.exit(1)
for name, path in [("melspectrogram", MEL_MODEL), ("embedding", EMB_MODEL),
                   ("wakeword_lite", LITE_MODEL), ("wakeword_full", FULL_MODEL)]:
    if not os.path.exists(path):
        print(f"模型未找到: {path}"); sys.exit(1)

# ========== 读取音频 ==========
with wave.open(AUDIO_FILE, 'rb') as wf:
    nch = wf.getnchannels()
    file_sr = wf.getframerate()
    nframes = wf.getnframes()
    audio_raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16)

print(f"音频: {os.path.basename(AUDIO_FILE)}")
print(f"  原始: {file_sr}Hz, {nch}ch, {nframes} 帧")

if nch > 1:
    audio_raw = audio_raw.reshape(-1, nch)[:, 0].copy()
if file_sr != SR:
    x_old = np.linspace(0, len(audio_raw)-1, len(audio_raw))
    x_new = np.linspace(0, len(audio_raw)-1, int(len(audio_raw)*SR/file_sr))
    audio_raw = np.interp(x_new, x_old, audio_raw.astype(np.float32)).astype(np.int16)

audio = audio_raw.astype(np.float32)
print(f"  处理后: {len(audio)} 样本, {len(audio)/SR:.2f}s")
print(f"  阈值: {THRESHOLD}, Gate阈值: {GATE_DEF}, Warmup: {ENABLE_WARMUP}")
print(f"  唤醒词: '{WAKE_WORD}'\n")

# ========== 加载 ONNX ==========
SESS_OPTS = ort.SessionOptions()
SESS_OPTS.inter_op_num_threads = 1; SESS_OPTS.intra_op_num_threads = 1
mel_sess  = ort.InferenceSession(MEL_MODEL,  SESS_OPTS)
emb_sess  = ort.InferenceSession(EMB_MODEL,  SESS_OPTS)
lite_sess = ort.InferenceSession(LITE_MODEL, SESS_OPTS)
full_sess = ort.InferenceSession(FULL_MODEL, SESS_OPTS)

# ========== 模拟 feed5Channel ==========
def feed5channel_mix(interleaved):
    """模拟 feed5Channel: 反交错 + 混合 mic 通道 0-3 -> mono"""
    n = len(interleaved) // 5
    mixed = np.zeros(n, dtype=np.float32)
    for i in range(n):
        base = i * 5
        s = (int(interleaved[base]) + int(interleaved[base+1]) +
             int(interleaved[base+2]) + int(interleaved[base+3]))
        mixed[i] = np.clip(s // 4, -32768, 32767)
    return mixed

# ========== 管线缓冲 ==========
raw_buf    = np.zeros(SR * 10, dtype=np.float32)
raw_pos    = 0; raw_len = 0; pending = 0
remainder  = np.array([], dtype=np.int16)

mel_buf    = np.ones((970, MEL_BINS), dtype=np.float32)
mel_head   = 0; mel_cnt = 76

feat_buf   = np.zeros((120, FEAT_DIM), dtype=np.float32)
feat_head  = 0; feat_cnt = 120

gate_score = 0.0; verifier_score = 0.0; warmup = 0

# ========== 流水线函数 ==========
def push_raw(data):
    global raw_pos, raw_len
    for v in data:
        raw_buf[raw_pos] = v; raw_pos = (raw_pos + 1) % len(raw_buf)
    raw_len = min(len(raw_buf), raw_len + len(data))

def tail_raw(count):
    count = min(count, raw_len)
    out = np.zeros(count, dtype=np.float32)
    start = (raw_pos - count + len(raw_buf)) % len(raw_buf)
    for i in range(count):
        out[i] = raw_buf[(start + i) % len(raw_buf)]
    return out

def run_mel(audio_data):
    out = mel_sess.run(None, {"input": audio_data.reshape(1, -1).astype(np.float32)})[0]
    flat = out.flatten()
    mf = len(flat) // MEL_BINS
    mel = np.zeros((mf, MEL_BINS), dtype=np.float32)
    for i in range(mf):
        off = i * MEL_BINS
        for j in range(MEL_BINS):
            mel[i, j] = flat[off + j] * MEL_SCALE + MEL_BIAS
    return mel

def push_mel(frames):
    global mel_head, mel_cnt
    for f in frames:
        mel_buf[mel_head] = f; mel_head = (mel_head + 1) % 970
        mel_cnt = min(970, mel_cnt + 1)

def slice_mel(start, end):
    if start < 0 or end > mel_cnt or start >= end:
        return None
    n = end - start
    out = np.zeros((n, MEL_BINS), dtype=np.float32)
    base = (mel_head - mel_cnt + start + 970) % 970
    for i in range(n):
        out[i] = mel_buf[(base + i) % 970]
    return out

def run_emb(mel_window):
    h, w = mel_window.shape
    out = emb_sess.run(None, {"input_1": mel_window.reshape(1, h, w, 1).astype(np.float32)})[0]
    return out.flatten()[:FEAT_DIM].astype(np.float32)

def push_feat(emb):
    global feat_head, feat_cnt
    feat_buf[feat_head] = emb; feat_head = (feat_head + 1) % 120
    feat_cnt = min(120, feat_cnt + 1)

def get_features(n):
    if feat_cnt < n: return None
    out = np.zeros((n, FEAT_DIM), dtype=np.float32)
    base = (feat_head + feat_cnt - n) % 120
    for i in range(n):
        out[i] = feat_buf[(base + i) % 120]
    return out

def infer(sess, features):
    inp = np.array(features[-FEAT_FRAMES:], dtype=np.float32).reshape(1, FEAT_FRAMES, FEAT_DIM)
    v = float(sess.run(None, {"input": inp})[0].flatten()[0])
    return max(0.0, min(1.0, v))

# ========== 主循环 ==========
print("=" * 60)
print("开始检测...")
print("=" * 60)

start_time = time.perf_counter()
frame_count = 0; detected = False; detect_time = None
all_g = []; all_v = []

i = 0
while i < len(audio):
    chunk = audio[i:i+CHUNK]
    if len(chunk) < CHUNK: break
    i += CHUNK

    # --- 模拟 feed5Channel ---
    chunk_i16 = chunk.astype(np.int16)
    interleaved = np.zeros(len(chunk) * 5, dtype=np.int16)
    for s in range(len(chunk)):
        b = s * 5
        v = int(chunk_i16[s])
        interleaved[b] = v; interleaved[b+1] = v
        interleaved[b+2] = v; interleaved[b+3] = v; interleaved[b+4] = 0
    mixed = feed5channel_mix(interleaved)

    # --- FeatureExtractor.feed() ---
    x = mixed.copy()
    if len(remainder) > 0:
        x = np.concatenate([remainder.astype(np.float32), x])
        remainder = np.array([], dtype=np.int16)

    total = int(pending + len(x))
    if total < CHUNK:
        push_raw(x); pending = total; continue

    rem = total % CHUNK
    if rem != 0:
        push_raw(x[:len(x)-rem])
        remainder = np.array(x[len(x)-rem:], dtype=np.int16)
        pending += len(x) - rem
    else:
        push_raw(x); pending += len(x)

    processed = 0
    if pending >= CHUNK and pending % CHUNK == 0:
        n_samples = int(pending)
        audio_seg = tail_raw(n_samples + MEL_EXTRA)
        new_mel = run_mel(audio_seg)
        push_mel(new_mel)

        audio_chunks = n_samples // CHUNK
        for c in range(audio_chunks - 1, -1, -1):
            offset = -MEL_STRIDE * c
            mel_end = mel_cnt if offset == 0 else mel_cnt + offset
            mel_start = int(mel_end - MEL_WIN)
            win = slice_mel(mel_start, mel_end)
            if win is None or win.shape[0] != MEL_WIN:
                continue
            emb = run_emb(win)
            push_feat(emb)

        processed = n_samples
        pending = 0

    # --- CascadeInference.infer() ---
    features = get_features(FEAT_FRAMES)
    if features is not None:
        frame_count += 1

        if ENABLE_WARMUP and warmup < 21:
            warmup += 1
            continue

        g = infer(lite_sess, features)
        gate_score = g
        v = 0.0
        if g >= GATE_DEF:
            v = infer(full_sess, features)
        verifier_score = v

        all_g.append(g); all_v.append(v)

        is_detect = v >= THRESHOLD
        if is_detect and not detected:
            detected = True
            detect_time = time.perf_counter() - start_time
            print(f"\n[检测到!] 唤醒词: '{WAKE_WORD}'")
            print(f"  detected        = True")
            print(f"  verifierScore   = {v:.4f}")
            print(f"  gateScore       = {g:.4f}")
            print(f"  threshold       = {THRESHOLD}")
            print(f"  检测时间         = {detect_time:.2f}s\n")

        if frame_count % 50 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  [{elapsed:6.2f}s] frame={frame_count:4d}  gate={g:.6f}  verifier={v:.6f}",
                  end="\r")

# ========== 汇总 ==========
elapsed = time.perf_counter() - start_time
print("\n\n" + "=" * 60)
print("检测汇总")
print("=" * 60)
print(f"  音频时长            : {len(audio)/SR:.2f}s")
print(f"  推理帧数            : {frame_count}")
print(f"  处理耗时            : {elapsed:.2f}s")
print(f"  实时率              : {len(audio)/SR/elapsed:.2f}x 实时" if elapsed > 0 else "N/A")
print(f"  唤醒结果            : {detected}")
print(f"  唤醒词              : '{WAKE_WORD}'")
print(f"  使用阈值            : {THRESHOLD}")
if all_v:
    print(f"  最高 gate 分数      : {np.max(all_g):.6f}")
    print(f"  最高 verifier 分数  : {np.max(all_v):.6f}")
    print(f"  平均 verifier 分数  : {np.mean(all_v):.6f}")
    det_count = sum(1 for s in all_v if s >= THRESHOLD)
    print(f"  检测帧数 (>={THRESHOLD}): {det_count}/{len(all_v)}")

print(f"\n-- Android SDK DetectionResult 等效输出 --")
print(f"  result.detected      = {detected}")
print(f"  result.verifierScore = {np.max(all_v) if all_v else 0:.6f}")
print(f"  result.gateScore     = {np.max(all_g) if all_g else 0:.6f}")
print(f"  result.threshold     = {THRESHOLD}")
print(f"  result.wakeWord      = \"{WAKE_WORD}\"")
if detected:
    print(f"  result.timestamp     = {detect_time:.2f}s")
