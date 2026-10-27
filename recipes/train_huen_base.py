import os, sys, dataclasses
from typing import Any, Dict, List, Union
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

HF_TOKEN = os.getenv("HF_TOKEN")

print("=== 1. Loading Processor and Model (Whisper Base 74M) ===")
model_id = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_id, language="english", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_id)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

print("=== 2. Loading Dataset ===")
dataset = load_dataset("benjaminogbonna/nigerian_accented_english_dataset")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print("Train rows:", len(dataset["train"]), "Val rows:", len(dataset["validation"]))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

print("=== 3. Mapping features and filtering max decoder positions (<448 tokens) ===")
train_ds = dataset["train"].select(range(min(1200, len(dataset["train"]))))
val_ds = dataset["validation"].select(range(min(200, len(dataset["validation"]))))

train_ds = train_ds.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=2)
val_ds = val_ds.map(prepare_dataset, remove_columns=dataset["validation"].column_names, num_proc=2)

# Strict Whisper architectural guard: decoder maximum length is 448 tokens
before_train = len(train_ds)
train_ds = train_ds.filter(lambda x: len(x["labels"]) < 448)
val_ds = val_ds.filter(lambda x: len(x["labels"]) < 448)
print(f"Filtered training samples exceeding 448 tokens: {before_train} -> {len(train_ds)}")

@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

output_dir = "/content/huen_base_out"
os.makedirs(output_dir, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=20,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=3,
    push_to_hub=True,
    hub_model_id="igunosalabs/Unose-Huen-Base",
    hub_token=HF_TOKEN,
    hub_strategy="checkpoint",
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

print("=== 4. Starting Full-Parameter Fine-Tuning (Whisper Base) ===")
trainer.train()

print("=== 5. Final Model Save and Push ===")
trainer.save_model(output_dir)
processor.save_pretrained(output_dir)
trainer.push_to_hub(commit_message="Production Unose-Huen-Base fine-tuning complete")
print("=== FINISHED HUEN-BASE ===")
