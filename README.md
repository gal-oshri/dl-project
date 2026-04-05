# Compressed Context Memory for Dialogue Summarization

Implementation of [Compressed Context Memory for Online Language Model Interaction](https://arxiv.org/abs/2312.03414) (ICLR 2024) applied to dialogue summarization.

The system compresses dialogue context into a single `<SUM>` token's KV cache using **Conditional LoRA**, enabling memory-efficient summarization.

## Architecture

```
Dialogue + <SUM> ──► Base LLM (frozen) + Conditional LoRA ──► KV Cache
                                                                  │
                                Extract <SUM> KV only             │
                                                                  ▼
                                                          <SUM> KV Cache
                                                                  │
                                                     Generate summary │
                                                                  ▼
                                                             Summary
```

### Key Components

| Component | Description |
|-----------|-------------|
| **SeparatedEmbedding** | Trainable embedding for `<SUM>` while base vocab is frozen |
| **ConditionalLoRA** | LoRA weights activate *only* at `<SUM>` token positions (Eq. 4 in paper) |
| **KV Extraction** | After encoding, only `<SUM>`'s KV cache is kept; dialogue KV is discarded |

## Project Structure

```
dl_project/
├── config.py                 # Dataclass configurations
├── model/
│   ├── embedding.py          # SeparatedEmbedding
│   ├── conditional_lora.py   # ConditionalLoRALinear + injection
│   ├── wrapper.py            # Model wrapper for input_ids propagation
│   └── builder.py            # Model assembly & weight loading
├── data/
│   ├── collator.py           # Compression data collator
│   └── dataset.py            # Dataset loading utilities
├── training/
│   └── trainer.py            # Training loop with KV extraction
├── inference/
│   └── generate.py           # Two-stage KV-based generation
├── train.py                  # Training entry point
├── infer.py                  # Inference entry point
├── requirements.txt
└── setup.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m dl_project train \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --dataset_name knkarthick/dialogsum \
    --output_dir ./compressed_context_model \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16
```

The training saves:
- `lora_weights.pt` — Conditional LoRA parameters
- `sum_embedding.pt` — `<SUM>` token embedding
- `training_info.json` — Training history
- Tokenizer files

## Inference

### Single dialogue
```bash
python -m dl_project infer \
    --weights_path ./compressed_context_model \
    --dialogue "Person1: Hi, how are you?\nPerson2: I'm fine, thanks."
```

### Evaluate on test set
```bash
python -m dl_project infer \
    --weights_path ./compressed_context_model \
    --dataset_eval \
    --num_examples 10
```

### Interactive mode
```bash
python -m dl_project infer --weights_path ./compressed_context_model
```

## How It Works

### Training (CCM Paper Algorithm 1)

1. **Encode**: Pass `dialogue + <SUM>` through the model with Conditional LoRA active only at `<SUM>` positions
2. **Extract**: Take only the `<SUM>` token's KV cache from the full cache
3. **Decode**: Feed summary tokens using `<SUM>` KV as the sole context (teacher forcing)
4. **Loss**: Cross-entropy on predicted summary tokens; gradients flow back through `<SUM>`'s KV

### Inference

1. **Compress**: Forward pass on `dialogue + <SUM>` → extract `<SUM>`'s KV (dialogue KV discarded)
2. **Generate**: Autoregressive generation using only the compressed KV cache

This achieves compression of the full dialogue into a single KV vector per layer (compression factor ≈ context_length / 1).

## Configuration

All defaults match the CCM paper (Tables 13-14):

| Parameter | Default | Paper Reference |
|-----------|---------|-----------------|
| LoRA rank | 8 | Table 14 |
| LoRA alpha | 16 | Table 14 |
| LoRA dropout | 0.05 | Table 14 |
| Target modules | q,k,v,o_proj | Table 14 |
| Learning rate | 3e-4 | Table 13 |
| Batch size | 4 | — |

## References

```bibtex
@inproceedings{kim2024compressed,
    title={Compressed Context Memory for Online Language Model Interaction},
    author={Kim, Jang-Hyun and Yeom, Junyoung and Yun, Sangdoo and Song, Hyun Oh},
    booktitle={ICLR},
    year={2024}
}
```

Original implementation: https://github.com/snu-mllab/context-memory
