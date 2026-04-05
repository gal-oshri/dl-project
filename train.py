"""
Training entry point for the Compressed Context Memory (CCM) summarization model.

Usage:
    python -m dl_project.train [OPTIONS]

    Options:
        --model_name        Base model (default: meta-llama/Llama-2-7b-chat-hf)
        --dataset_name      HF dataset (default: knkarthick/dialogsum)
        --output_dir        Save directory (default: ./compressed_context_model)
        --batch_size        Training batch size (default: 4)
        --num_epochs        Number of training epochs (default: 3)
        --learning_rate     Learning rate (default: 3e-4)
        --lora_r            LoRA rank (default: 8)
        --lora_alpha        LoRA alpha (default: 16)
        --max_input_length  Max dialogue token length (default: 400)
        --max_target_length Max summary token length (default: 100)
"""

import argparse
import json
import logging
import os
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CCM compression adapter")

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--dataset_name", type=str, default="knkarthick/dialogsum")
    parser.add_argument("--output_dir", type=str, default="./compressed_context_model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_input_length", type=int, default=400)
    parser.add_argument("--max_target_length", type=int, default=100)
    parser.add_argument(
        "--lora_targets",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for gated models (alternative to huggingface-cli login)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.hf_token:
        from huggingface_hub import login
        print("Logging in to Hugging Face with provided token...")
        login(token=args.hf_token)

    from dl_project.config import CCMConfig, ModelConfig, LoRAConfig, DataConfig, TrainingConfig
    from dl_project.model.builder import build_model
    from dl_project.data.dataset import load_dialogsum, create_dataloaders
    from dl_project.data.collator import CompressionDataCollator
    from dl_project.training.trainer import run_training

    config = CCMConfig(
        model=ModelConfig(model_name=args.model_name),
        lora=LoRAConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_targets,
        ),
        data=DataConfig(
            dataset_name=args.dataset_name,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
        ),
    )

    logger.info("Building model: %s", config.model.model_name)
    model, tokenizer, compression_token_id = build_model(config)

    logger.info("Loading dataset: %s", config.data.dataset_name)
    dataset = load_dialogsum(config.data.dataset_name)

    collator = CompressionDataCollator(
        tokenizer=tokenizer,
        compression_token_id=compression_token_id,
        max_input_length=config.data.max_input_length,
        max_target_length=config.data.max_target_length,
    )
    train_dl, val_dl, _ = create_dataloaders(dataset, collator, config.training.batch_size)

    logger.info("Starting training...")
    history = run_training(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        compression_token_id=compression_token_id,
        tokenizer=tokenizer,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        output_dir=config.training.output_dir,
    )

    info_path = os.path.join(config.training.output_dir, "training_info.json")
    with open(info_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training complete. History saved to %s", info_path)


if __name__ == "__main__":
    main()
