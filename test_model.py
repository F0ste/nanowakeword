"""Test the trained "hello Ava" wake word model."""
import sys
sys.path.insert(0, ".")

from nanowakeword import NanoInterpreter

# Load the distilled (lite) model
model_path = "trained_models/hi_hotel_multilang_v1/model/hi_hotel_multilang_v1_lite.onnx"
interpreter = NanoInterpreter.load_model(model_path)

print(f"Model loaded: {model_path}")
print(f"Listening for 'hello Ava'... Press Ctrl+C to stop.\n")

try:
    interpreter.listen(
        threshold=0.5,
        on_detection=lambda name, score: print(f"\n>>> DETECTED '{name}' score={score:.4f}"),
        on_score=lambda vs, gs: print(".", end="", flush=True)
    )
except KeyboardInterrupt:
    print("\nStopped.")
