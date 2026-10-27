import argparse
import time
from faster_whisper import WhisperModel

def main():
    parser = argparse.ArgumentParser(description="Evaluate audio transcription latency and output.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (WAV/MP3/OGG)")
    parser.add_argument("--model", type=str, default="igunosalabs/Unose-Huen-Small", help="Hugging Face model ID or path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--compute_type", type=str, default="int8", help="Quantization type (int8, float16, float32)")
    args = parser.parse_args()

    print(f"Loading model: {args.model} ({args.compute_type}) on {args.device}...")
    t0 = time.time()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    print(f"Transcribing {args.audio}...")
    t1 = time.time()
    segments, info = model.transcribe(args.audio, beam_size=5, language="en")
    
    text_segments = []
    for s in segments:
        print(f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}")
        text_segments.append(s.text.strip())
    
    elapsed = time.time() - t1
    full_text = " ".join(text_segments)
    print("=" * 60)
    print(f"Transcription completed in {elapsed:.2f}s")
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    print(f"Total words: {len(full_text.split())}")
    print("Full Transcript:")
    print(full_text)

if __name__ == "__main__":
    main()
