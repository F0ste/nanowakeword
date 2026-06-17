"""
双塔级联唤醒词检测测试
======================

架构说明：
  第一级（门控/粗筛）：lite 模型 — 轻量快速，逐帧运行，输出门控置信度 (gate_score)
  第二级（验证/确认）：full 模型 — 仅在 gate_score >= gate_threshold 时触发，输出验证置信度 (verifier_score)

工作流程：
  音频帧 → lite 模型 (门控置信度) ──低于阈值──→ 跳过 (SKIP)
                                    │
                                    └──高于阈值──→ full 模型 (确认中) ──≥ 检测阈值──→ 触发唤醒
                                                                      │
                                                                      └──< 检测阈值──→ 忽略

优势：
  - lite 模型计算量小，持续运行不费资源
  - full 模型只在必要时介入确认，大幅降低整体计算量
  - 兼顾了低功耗与高精度
"""
import sys
sys.path.insert(0, ".")

from nanowakeword import NanoInterpreter

# ============================================================================
# 配置区 — 根据实际路径修改
# ============================================================================
MODEL_PATH = "trainedmodel/new212/model/hi_hotel_multilang_10k.onnx"
GATE_THRESHOLD = 0.3    # 门控阈值：lite 置信度超过此值才触发验证模型
DETECT_THRESHOLD = 0.4  # 检测阈值：验证置信度超过此值才视为真正的唤醒

# ============================================================================
# 加载双塔模型 (cascade=True 启用级联模式)
# ============================================================================
interpreter = NanoInterpreter.load_model(
    MODEL_PATH,
    cascade=True,
    gate_threshold=GATE_THRESHOLD,
)

print(f"级联模式已启用")
print(f"  门控模型 (lite):  {interpreter.gate_name}")
print(f"  验证模型 (full):  {interpreter.model_name}")
print(f"  门控阈值: {GATE_THRESHOLD}  |  检测阈值: {DETECT_THRESHOLD}")
print(f"\n正在监听 'hello Ava'... 按 Ctrl+C 停止\n")

# ============================================================================
# 实时监听
# ============================================================================
try:
    interpreter.listen(
        threshold=DETECT_THRESHOLD,
        on_detection=lambda name, score: print(
            f"\n>>> 【唤醒】检测到 '{name}'  验证置信度={score:.4f}"
        ),
        on_score=lambda vs, gs: print(
            f"\r门控置信度={gs:.3f}  验证置信度={vs:.4f}"
            + (" [跳过]" if gs < GATE_THRESHOLD else " [确认中...]"),
            end="", flush=True
        ),
    )
except KeyboardInterrupt:
    print("\n已停止。")
