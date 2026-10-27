# Unose-Huen: Nigerian-Accented Speech-to-Text Suite

**Unose-Huen** is an Automatic Speech Recognition (ASR) suite fine-tuned to improve transcription accuracy on Nigerian English accents, conversational cadence, and colloquial prosody.

Developed by **Igunosa Labs**, Unose-Huen provides fine-tuned Whisper checkpoints optimized for edge and server deployment.

---

## Models

| Model | Parameters | Target | Validation WER | Hugging Face Hub |
| :--- | :--- | :--- | :--- | :--- |
| **Unose-Huen-Base** | 74M | CPU / GPU / edge | **48.38%** | [igunosalabs/Unose-Huen-Base](https://huggingface.co/igunosalabs/Unose-Huen-Base) |
| **Unose-Huen-Small** | 244M | CPU / GPU / edge | **37.07%** | [igunosalabs/Unose-Huen-Small](https://huggingface.co/igunosalabs/Unose-Huen-Small) |

*Evaluated on the Nigerian Accented English validation benchmark (`benjaminogbonna/nigerian_accented_english_dataset`).*

---

## Precision Formats

Each release on Hugging Face contains weights across multiple precisions:

- `model.safetensors`: Full precision PyTorch weights for Hugging Face Transformers.
- `ctranslate2_float32/`: 32-bit floating point CTranslate2 format.
- `ctranslate2_float16/`: 16-bit floating point CTranslate2 format.
- `ctranslate2_int8_float16/`: Mixed precision (Int8 weights with FP16 activations).
- `ctranslate2_int8/`: 8-bit quantized format for lightweight edge and CPU deployment.

---

## Environment Setup

### Prerequisites
- Python 3.10+
- FFmpeg (for audio pre-processing)

```bash
# Clone the repository
git clone https://github.com/Igunosa-Labs/unose-huen.git
cd unose-huen

# Install dependencies
pip install faster-whisper transformers torch datasets evaluate jiwer
```

---

## Inference

### Using `faster-whisper` (Recommended for CPU & Edge)

```python
from faster_whisper import WhisperModel

# Load Unose-Huen-Small in int8 quantization on CPU
model = WhisperModel(
    "igunosalabs/Unose-Huen-Small",
    device="cpu",
    compute_type="int8",
    subfolder="ctranslate2_int8"
)

segments, info = model.transcribe("audio.wav", beam_size=5, language="en")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### Using PyTorch `transformers`

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

model_id = "igunosalabs/Unose-Huen-Small"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe("audio.wav")
print(result["text"])
```

---

## Running Benchmarks

Evaluate transcription speed and output for an audio file:

```bash
python3 benchmark/benchmark_head_to_head.py --audio sample.wav --model igunosalabs/Unose-Huen-Small --device cpu --compute_type int8
```

---

## Training Recipes

Training scripts are located in `recipes/`:

- `recipes/train_huen_base.py`: Fine-tuning recipe for Whisper Base.
- `recipes/train_huen_small.py`: Fine-tuning recipe for Whisper Small.

Both recipes enforce strict sequence filtering (`len(labels) < 448`) to adhere to Whisper's decoder positional limit.

To launch training:

```bash
export HF_TOKEN="your_huggingface_token"
python3 recipes/train_huen_small.py
```

---

## License

Developed by **Igunosa Labs**. Apache-2.0 License.
Base Whisper architecture copyright OpenAI.
