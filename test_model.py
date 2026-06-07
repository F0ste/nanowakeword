"""Test the cascade (dual-tower) wake word detection.

Stage 1 (gate):   lite model  - fast, runs every frame
Stage 2 (verifier): full model - runs only when gate_score >= gate_threshold
"""
import sys
sys.path.insert(0, ".")

from nanowakeword import NanoInterpreter

model_path = "trained_models/hi_hotel_multilang_10k/model/hi_hotel_multilang_10k.onnx"
interpreter = NanoInterpreter.load_model(model_path, cascade=True, gate_threshold=0.3)

print(f"Cascade mode: gate='{interpreter.gate_name}' -> verifier='{interpreter.model_name}'")
print(f"Listening for 'hello Ava'... Press Ctrl+C to stop.\n")

try:
    interpreter.listen(
        threshold=0.5,
        on_detection=lambda name, score: print(
            f"\n>>> DETECTED '{name}' verifier_score={score:.4f}"
        ),
        on_score=lambda vs, gs: print(
            f"\rg={gs:.3f} v={vs:.4f}" + (" [SKIP]" if gs < 0.3 else " [RUN]"),
            end="", flush=True
        ),
    )
except KeyboardInterrupt:
    print("\nStopped.")
